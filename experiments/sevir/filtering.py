import os
import wandb
import argparse
import numpy as np
import uuid
import torch
from datetime import datetime
from pathlib import Path
from typing import Tuple
from netCDF4 import Dataset as NetCDFDataset
from sda.score import (
    GaussianScore, VPSDE,
    DPSGaussianScore, MMPSGaussianScore,
    MCScoreNet, ScoreUNet
)
from utils import ACTIVATIONS, SEVIRLightningDataModule, save_video
from metrics import rmse, crps_ens, spread_skill_ratio


class LocalScoreUNet(ScoreUNet):
    """Score U-Net with forcing channel (matches your training)."""

    def __init__(self, channels: int, size: int, periodic: bool = False, **kwargs):
        super().__init__(channels, 1, periodic=periodic, **kwargs)

        domain = 2 * torch.pi / size * (torch.arange(size) + 0.5)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()
        self.register_buffer("forcing", forcing)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        return super().forward(x, t, self.forcing)


def make_score(
    size: int,
    window: int,
    embedding: int,
    hidden_channels: Tuple[int],
    hidden_blocks: Tuple[int],
    kernel_size: int,
    activation: str,
    periodic: bool = False,
    **absorb
) -> torch.nn.Module:
    score = MCScoreNet(2, order=window // 2)
    score.kernel = LocalScoreUNet(
        size=size,
        channels=window,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        periodic=periodic,
        padding_mode='circular',
    )
    return score


def parse_args():
    p = argparse.ArgumentParser(
        "Run inference for SDA (parity with notebook).")
    # Data
    p.add_argument("--data_dir", type=str, required=True)  # specify
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    # Obs + guidance
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--obs_pct", type=float, default=0.1)
    p.add_argument("--obs_sigma", type=float, default=0.001)
    p.add_argument("--init_sigma", type=float, default=0.001)
    p.add_argument("--fixed_obs", action="store_true")
    p.add_argument("--n_ens", type=int, default=20)
    p.add_argument("--guidance_method", type=str,
                   choices=["DPS", "DPS_scale", "MMPS"], default="DPS")
    p.add_argument("--guidance_strength", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--corrections", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.5)
    # Checkpoint + output
    p.add_argument("--ckpt_path", type=str, required=True)  # specify
    p.add_argument("--output_dir", type=str, default="./output")
    # Logging
    p.add_argument("--wandb_project", type=str, default="ScoreDA_SQG")
    p.add_argument("--wandb_entity", type=str, default="stima")
    # Flags
    p.add_argument("--initial_condition", action="store_true",
                   help="Add initial condition.")
    p.add_argument('--data_index', type=int, default=0,
                   help='Index of the data sample to use for assimilation')
    p.add_argument('--start_time', type=int, default=0,
                   help='Start time index for assimilation (default: 0)')
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    torch.set_default_dtype(torch.float32)

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    dm = SEVIRLightningDataModule(
        seq_len=25,
        sample_mode='sequent',
        stride=6,
        batch_size=100,
        layout='NTCHW',
        output_type=np.float32,
        preprocess=True,
        rescale_method="01",
        verbose=False,
        aug_mode='0',  # NOTE: No augmentation on the data
        ret_contiguous=False,
        # datamodule_only
        dataset_name='sevirlr',
        start_date=None,
        train_test_split_date=[2019, 6, 1],
        end_date=None,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        sevir_dir=args.data_dir,
    )
    dm.setup()

    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()

    # NOTE: To ensure we use the same validation samples as in the other experiments
    ns = [9, 10, 28, 49, 58, 59, 60, 81, 91, 96]
    val_data = val_loader.dataset[ns[args.data_index]]

    val_data = torch.tensor(val_data, device=device)[args.start_time:]
    T, C, H, W = val_data.shape

    # Create observation mask and corresponding indices
    mask = (torch.rand(H, W, device=device) < args.obs_pct).float()
    iy, ix = (mask > 0.5).nonzero(as_tuple=True)

    # Load score model
    ckpt = torch.load(args.ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    if not cfg:
        raise KeyError("Checkpoint missing 'config'.")

    window = int(cfg["window"])
    embedding = int(cfg["embedding"])
    hidden_channels = tuple(cfg["hidden_channels"])
    hidden_blocks = tuple(cfg["hidden_blocks"])
    kernel_size = int(cfg["kernel_size"])
    activation = str(cfg["activation"])
    periodic = bool(cfg.get("periodic", False))

    CONFIG = dict(
        size=H,
        window=window,
        embedding=embedding,
        hidden_channels=tuple(hidden_channels),
        hidden_blocks=tuple(hidden_blocks),
        kernel_size=kernel_size,
        activation=activation,
        periodic=periodic,
    )
    score = make_score(**CONFIG).to(device).eval()
    state = ckpt.get("model_state", None)
    score.kernel.load_state_dict(state, strict=True)

    def A(x):  # Masking + initial condition
        # x has shape (B, T, C, H, W)
        B, L, C, H, W = x.shape
        return torch.concatenate([x[:, 0].flatten(start_dim=1),  # Identity map on t=0:5 to impose initial condition
                                  x[:, 1:args.window, :, iy, ix].reshape(
                                      B, -1)  # Sparse observations for t > 0
                                  ], dim=1)

    ys = val_data[..., iy, ix]
    filter_window = T

    y = A(val_data[None])
    obs_sigma_full = args.obs_sigma * torch.ones_like(y)
    N = H*W  # NOTE: Could add the window -1 here.
    obs_sigma_full[:, :N] = args.init_sigma**2
    eps_full = torch.randn_like(y)
    y = y + obs_sigma_full * eps_full

    # Set up guided score SDE
    if args.guidance_method == 'DPS':
        guided_eps = DPSGaussianScore(
            y, A=A, std=args.obs_sigma,
            sde=VPSDE(score, shape=()),
            gamma=args.gamma,
            scale=False,
            guidance_strength=args.guidance_strength,
        )
    elif args.guidance_method == 'DPS_scale':
        guided_eps = DPSGaussianScore(
            y, A=A, std=args.obs_sigma,
            sde=VPSDE(score, shape=()),
            gamma=args.gamma,
            scale=True,
            guidance_strength=args.guidance_strength,
        )
    elif args.guidance_method == 'MMPS':
        guided_eps = MMPSGaussianScore(
            y,
            observation_fn=A,
            std=args.obs_sigma,
            init_std=args.init_sigma,
            sde=VPSDE(score, shape=()),
            guidance_strength=args.guidance_strength,
        )

    guided_sde = VPSDE(
        eps=guided_eps,
        shape=(args.window, C, H, W),
    ).to(device)

    # Sample at inital time
    posterior_samples = guided_sde.sample((args.n_ens,),
                                          steps=args.steps,
                                          corrections=args.corrections,
                                          tau=args.tau)

    state = posterior_samples[:, [-1], ...]
    run_name = f"SEVIR_SDA_filtering_{args.obs_pct}pct_{args.guidance_method}_strength{args.guidance_strength}_em_steps{args.steps}_corr{args.corrections}_data{args.data_index}"

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=run_name,
               config=vars(args))

    pred_t = state.squeeze(1).detach().cpu()                 # (n_ens,C,H,W)
    gt_t = val_data[args.window-1].detach().cpu().float()  # (C,H,W)
    wandb.log({
        "RMSE": rmse(torch.mean(pred_t, dim=0), gt_t).item(),
        "CRPS": crps_ens(pred_t, gt_t, ens_dim=0).item(),
        "Spread Skill Ratio": spread_skill_ratio(pred_t, gt_t, ens_dim=0).item(),
    }, step=args.window-1+args.start_time)

    filtered_states = [state]
    init_std = posterior_samples.std(dim=0).mean(dim=[1, 2, 3])[1]
    for i in range(1, filter_window-args.window+1):
        ic_vec = posterior_samples[:, 1].flatten(start_dim=1)
        y_clean = torch.cat([ic_vec, ys[None, i+1:i+args.window].reshape(
            1, -1).repeat(args.n_ens, 1)], dim=1)  # Shape (B, M_full)

        obs_sigma_full = args.obs_sigma * torch.ones_like(y_clean)
        obs_sigma_full[:, :N] = init_std
        # fixed noise reused by prefixing
        eps_full = torch.randn_like(y_clean)
        y = y_clean + obs_sigma_full * eps_full

        # guidance
        if args.guidance_method == 'DPS':
            guided_sde = VPSDE(
                DPSGaussianScore(
                    y,
                    A=A,
                    std=args.obs_sigma,
                    sde=VPSDE(score, shape=()),
                    gamma=1e-2,
                    guidance_strength=0.1,
                    scale=False,
                ),
                shape=(args.window, C, H, W)
            ).cuda()
        elif args.guidance_method == 'MMPS':
            guided_sde = VPSDE(
                MMPSGaussianScore(
                    y,
                    observation_fn=A,
                    std=args.obs_sigma,
                    # init_std=None,
                    init_std=init_std,
                    sde=VPSDE(score, shape=()),
                    guidance_strength=1.0,
                    solver='gmres',
                    iterations=1,
                ),
                shape=(args.window, C, H, W)
            ).cuda()

        posterior_samples = guided_sde.sample((args.n_ens,),
                                              steps=100,
                                              corrections=1,
                                              tau=0.5)

        state = posterior_samples[:, [-1], ...]
        init_std = posterior_samples.std(dim=0).mean(dim=[1, 2, 3])[1]
        filtered_states.append(state)

        # ---- Per-iteration metrics: log for time t = i + window ----
        t_idx = i + args.window - 1
        pred_t = state.squeeze(1).detach().cpu()  # (n_ens,C,H,W)
        gt_t = val_data[t_idx].detach().cpu().float()  # (C,H,W)
        step_idx = t_idx + args.start_time
        wandb.log({
            "RMSE": rmse(torch.mean(pred_t, dim=0), gt_t).item(),
            "CRPS": crps_ens(pred_t, gt_t, ens_dim=0).item(),
            "Spread Skill Ratio": spread_skill_ratio(pred_t, gt_t, ens_dim=0).item(),
        }, step=step_idx)

    filtered_states = torch.cat(
        filtered_states, dim=1).cpu()  # (n_ens, T, C, H, W)
    posterior_time_first = torch.swapaxes(
        filtered_states, 0, 1).numpy()  # (T, n_ens, C, H, W)

    # ---- Save NetCDF ----
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    random_id = str(uuid.uuid4())[:4]  # Use first 4 characters of UUID
    timestamp = datetime.now().strftime("%m_%d_%H")
    file_name = f"SDA_SEVIR_run_{args.data_index}_{timestamp}_{random_id}"
    nc_filename = Path(args.output_dir) / f"{file_name}.nc"
    nc = NetCDFDataset(nc_filename, mode="w", format="NETCDF4_CLASSIC")
    nc.createDimension("x", W)
    nc.createDimension("y", H)
    nc.createDimension("z", C)
    nc.createDimension("t", T-args.window+1)
    nc.createDimension("ens", args.n_ens)
    ground_truth = nc.createVariable(
        "ground_truth", np.float32, ("t", "z", "y", "x"), zlib=True)
    x_assim = nc.createVariable(
        "x_assim",      np.float32, ("t", "ens", "z", "y", "x"), zlib=True)
    ground_truth[:] = val_data[args.window-1:].cpu().numpy()
    x_assim[:] = posterior_time_first
    nc.sync()
    nc.close()

    # ---- Save animation ----
    vid_path = save_video(
        torch.tensor(posterior_time_first).float().cpu(),
        (val_data[args.window-1:]
         .clone().detach().float()).cpu(),
        args.obs_sigma, args.obs_pct, obs_fn=lambda x: x, level=0
    )
    wandb.log({"animation": wandb.Video(vid_path, fps=10, format="mp4")})
    wandb.finish()


if __name__ == "__main__":
    main()
