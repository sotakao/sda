import os
import argparse
import torch
import numpy as np
from torch import Tensor
from typing import *
from pathlib import Path
from netCDF4 import Dataset as NetCDFDataset
from sda.score import GaussianScore, GaussianScoreScaled, DPSGaussianScore, VPSDE, ScaledVPSDE, MCScoreNet, ScoreUNet
from utils import TrajectoryDataset, save_spectrum, save_video, OBS_FNS, ACTIVATIONS
import wandb
from metrics import rmse, crps_ens, spread_skill_ratio

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class LocalScoreUNet(ScoreUNet):
    r"""Creates a score U-Net with a forcing channel."""

    def __init__(
        self,
        channels: int,
        size: int = 64,
        **kwargs,
    ):
        super().__init__(channels, 1, **kwargs)

        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.forcing)


def make_score(
    window: int,
    embedding: int,
    hidden_channels: Tuple[int],
    hidden_blocks: Tuple[int],
    kernel_size: int,
    activation: str,
    **absorb
) -> torch.nn.Module:
    
    score = MCScoreNet(2, order=window // 2)

    score.kernel = LocalScoreUNet(
        channels=window * 2,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        padding_mode='circular',
    )
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for SDA.")
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data.")
    parser.add_argument("--train_file", type=str, default="sqg_pv_train.h5", help="Training dataset filename.")
    parser.add_argument("--hrly_freq", type=int, default=3, help="Frequency of hourly data.")
    # Model settings
    parser.add_argument("--window", type=int, default=5, help="SDA Markov blanket size.")
    parser.add_argument("--embedding", type=int, default=64, help="Embedding size for the model.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[96, 192, 384], help="Hidden channels for the model.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[3, 3, 3], help="Hidden blocks for the model.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the model.")
    parser.add_argument("--activation", type=str, default="SiLU", help="Activation function for the model.")
    parser.add_argument("--epochs", type=int, default=4096, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for the optimizer.")
    # Obs+Guidance settings
    parser.add_argument("--obs_type", type=str, choices=["grid", "random"], default="random", help="Observation type.")
    parser.add_argument("--obs_stride", type=int, default=4, help="Observation stride (for grid).")
    parser.add_argument("--obs_pct", type=float, default=0.05, help="Observation percentage (for random).")
    parser.add_argument("--obs_fn", type=str, choices=["linear", "arctan15"], default="linear", help="Observation function.")
    parser.add_argument("--obs_sigma", type=float, default=0.3, help="Observation noise std.")
    parser.add_argument("--fixed_obs", action="store_true", help="Use a fixed observation mask across samples.")
    parser.add_argument("--n_ens", type=int, default=20, help="Number of ensemble members.")
    parser.add_argument("--guidance_method", type=str, choices=["DPS"], default="DPS", help="Guidance method.")
    parser.add_argument("--gamma", type=float, default=1e-2, help="Gamma term in guidance.")
    parser.add_argument("--steps", type=int, default=100, help="Number of sampling steps.")
    parser.add_argument("--corrections", type=int, default=1, help="Number of correction steps.")
    parser.add_argument("--tau", type=float, default=0.5, help="The amplitude of Langevin steps.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs.")
    # Logging arguments
    parser.add_argument("--log_wandb", type=int, default=1, help="Log results to WandB or not.")
    parser.add_argument("--wandb_project", type=str, default="ScoreDA_SQG", help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, default="stima", help="WandB entity name.")
    return parser.parse_args()

def main():
    args = parse_args()
    args.log_wandb = bool(args.log_wandb)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    if args.log_wandb:
        exp_name = f"SDA_window{args.window}_{args.obs_fn}_{args.obs_pct}pct_{args.guidance_method}_gamma{args.gamma}_steps{args.steps}_corrections{args.corrections}_tau{args.tau}"
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=exp_name,
                   config=vars(args))

    # Load trainset only to get normalization stats (mean/std)
    # trainset = TrajectoryDataset(
    #     Path(args.data_dir) / f"{args.hrly_freq}hrly" / args.train_file,
    #     normalize=True,
    #     window=args.window,
    #     flatten=True,
    # )
    # pv_mean = torch.as_tensor(trainset.mean, device=device, dtype=torch.float32)
    # pv_std = torch.as_tensor(trainset.std, device=device, dtype=torch.float32)
    pv_mean = torch.as_tensor(0.0, device=device, dtype=torch.float32)
    pv_std = torch.as_tensor(2672.232, device=device, dtype=torch.float32)

    # Observation function
    if args.obs_fn not in OBS_FNS:
        raise ValueError(f"obs_fn '{args.obs_fn}' not in {list(OBS_FNS.keys())}")
    obs_fn = OBS_FNS[args.obs_fn]

    # RNG to match assimilate.py (seed=42)
    rng = np.random.RandomState(42)

    # Load pv_truth from NetCDF (test file)
    test_file = Path(args.data_dir) / f"{args.hrly_freq}hrly" / f"sqg_N64_{args.hrly_freq}hrly_100.nc"
    nc_truth = NetCDFDataset(test_file, 'r')
    pv_truth_nc = nc_truth.variables['pv']  # expected shape: (T, 2, ny, nx)
    T, Z, ny, nx = pv_truth_nc.shape

    # Slice window and move to torch
    pv_truth = torch.tensor(np.array(pv_truth_nc[:T, ...]), dtype=torch.float32, device=device)  # (T, 2, ny, nx)
    scalefact = nc_truth.f*nc_truth.theta0/nc_truth.g # 0.003061224412462883

    # Optional: close file handle after reading metadata and pv
    nc_truth.close()

    # Build observation mask
    def make_obs_mask(ny_, nx_):
        if args.obs_type == "grid":
            stride = args.obs_stride
            mask = torch.zeros(ny_, nx_, device=device)
            mask[::stride, ::stride] = 1.0
        elif args.obs_type == "random":
            nobs = int(ny_ * nx_ * args.obs_pct)
            idx = rng.choice(ny_ * nx_, nobs, replace=False)
            mask = torch.zeros(ny_ * nx_, device=device)
            mask[torch.from_numpy(idx).to(device)] = 1.0
            mask = mask.view(ny_, nx_)
        else:
            raise NotImplementedError("Unsupported observation type.")
        return mask  # (ny, nx)

    obs_mask = make_obs_mask(ny, nx)  # (ny, nx)

    # Load checkpointed score/SDE
    ckpt_dir = Path("../../runs_sqg") / f"mcscore_vpsde_sqg_window_{args.window}" / "checkpoints"
    files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}.")
    payload = torch.load(files[-1], map_location=device)

    CONFIG = {
        'window': args.window,
        'embedding': args.embedding,
        'hidden_channels': tuple(args.hidden_channels),
        'hidden_blocks': tuple(args.hidden_blocks),
        'kernel_size': args.kernel_size,
        'activation': args.activation,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
    }

    # Build model and SDE
    score_obj = make_score(**CONFIG)
    score_obj.load_state_dict(payload["score_state"])
    score_obj = score_obj.to(device).eval()

    # Observation operator: normalized model state -> physical -> obs space, masked
    def A_model(x):
        # x_norm: (T, 2, ny, nx)
        # x_phys = x_norm * pv_std + pv_mean
        return obs_fn(x) * obs_mask  # broadcast on (ny, nx)

    # Create observations from scaled pv_truth with noise (only at observed points)
    noise = torch.randn_like(pv_truth) * args.obs_sigma
    y_star = obs_fn(scalefact * pv_truth) * obs_mask + noise * obs_mask  # (T, 2, ny, nx)
    # y_star = y_star.unsqueeze(0) # (1, T, 2, ny, nx) for broadcasting in score

    # Guided SDE
    if args.guidance_method.lower() == "dps":
        scale = (scalefact * pv_std).to(device)
        score_scaled = lambda x, t, c: score_obj(x/scale, t, c)
        guided_sde = ScaledVPSDE(
                  GaussianScore(y_star,
                                A=A_model,
                                std=args.obs_sigma,
                                sde=ScaledVPSDE(score_scaled, shape=(), scale=scale),
                                gamma=args.gamma,
                                ),
            scale=scale,
            shape=y_star.shape,
        ).to(device)
    else:
        raise NotImplementedError("Unsupported guidance method.")

    # Sample posterior ensemble sequentially to reduce GPU memory
    posterior_list = []
    for e in range(args.n_ens):
        print(f"Sampling ensemble member {e+1}/{args.n_ens}...")
        sample = guided_sde.sample((1,),
                                   steps=args.steps,
                                   corrections=args.corrections,
                                   tau=args.tau)  # (1, T, 2, ny, nx)
        posterior_list.append(sample.squeeze(0).cpu())      # (T, 2, ny, nx)
        # free GPU mem between members
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # (T, ens, 2, ny, nx) for saving/logging
    posterior_time_first = torch.stack(posterior_list, dim=1).numpy()

    # Prepare NetCDF output (t dimension equals window length)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    nc_filename = output_dir / "inference_results.nc"
    nc = NetCDFDataset(nc_filename, mode="w", format="NETCDF4_CLASSIC")
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)
    nc.createDimension("z", 2)
    nc.createDimension("t", T)
    nc.createDimension("ens", args.n_ens)
    ground_truth = nc.createVariable("ground_truth", np.float32, ("t", "z", "y", "x"), zlib=True)
    x_assim = nc.createVariable("x_assim", np.float32, ("t", "ens", "z", "y", "x"), zlib=True)

    ground_truth[:] = pv_truth.cpu().numpy()                # (T, 2, ny, nx)
    x_assim[:] = posterior_time_first                       # (T, ens, 2, ny, nx)
    nc.sync(); nc.close()

    # Log metrics to WandB per time step (compute in PV units)
    if args.log_wandb:
        pv_mean_s = pv_mean.detach().cpu()
        pv_std_s = pv_std.detach().cpu()
        for t in range(T):
            pred_t = torch.tensor(posterior_time_first[t]).float()      # (ens, 2, ny, nx) normalized
            pred_phys = pred_t * pv_std_s + pv_mean_s                   # de-normalize to PV units
            gt_t = pv_truth[t].detach().cpu().float()                   # (2, ny, nx) PV units
    
            wandb.log({
                "RMSE": rmse(torch.mean(pred_phys, dim=0), gt_t).item(),
                "CRPS": crps_ens(pred_phys, gt_t, ens_dim=0).item(),
                "Spread Skill Ratio": spread_skill_ratio(pred_phys, gt_t, ens_dim=0).item(),
            }, step=t)
    
        # Log to WandB: spectrum at a representative time and an animation
        t_mid = T // 2
    
        # save_spectrum expects forecast (ens, 2, H, W) and truth (1, 2, H, W)
        spec_path = save_spectrum(
            torch.tensor(posterior_time_first[t_mid]).float(),         # (ens, 2, ny, nx)
            pv_truth[t_mid].clone().detach().unsqueeze(0).float(),        # (1, 2, ny, nx)
            time=t_mid
        )
        wandb.log({"spectrum": wandb.Image(spec_path)}, step=t_mid)
    
        vid_path = save_video(
            torch.tensor(posterior_time_first).float(),                # (T, ens, 2, ny, nx) -> function handles
            pv_truth.clone().detach().float(),                            # (T, 2, ny, nx)
            args.obs_sigma,
            args.obs_pct,
            obs_fn=obs_fn,
            level=0
        )
        wandb.log({"animation": wandb.Video(vid_path, fps=10, format="mp4")})
        wandb.finish()

    else:
        print(f"RMSE: {rmse(torch.mean(pred_phys, dim=0), gt_t).item()}")
        print(f"CRPS: {crps_ens(pred_phys, gt_t, ens_dim=0).item()}")
        print(f"Spread Skill Ratio: {spread_skill_ratio(pred_phys, gt_t, ens_dim=0).item()}")

if __name__ == "__main__":
    main()

