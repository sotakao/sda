import os
import wandb
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from torch import Tensor
from netCDF4 import Dataset as NetCDFDataset
from sda.score import (
    GaussianScore, ScaledVPSDE,
    DPSGaussianScore, MMPSGaussianScore,
    MCScoreNet, ScoreUNet
)
from utils import OBS_FNS, ACTIVATIONS, save_spectrum, save_video
from metrics import rmse, crps_ens, spread_skill_ratio

from datetime import datetime
import uuid


# ---------------------------
#  Model + score definitions
# ---------------------------
class LocalScoreUNet(ScoreUNet):
    """Score U-Net with a fixed forcing channel (same as notebook)."""

    def __init__(self, channels: int, size: int = 64, **kwargs):
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


# ---------------------------
#  CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Run inference for SDA (parity with notebook).")
    # Data
    # p.add_argument("--val_file", type=str, required=True)
    # Model
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--embedding", type=int, default=64)
    p.add_argument("--hidden_channels", type=int,
                   nargs="+", default=[96, 192, 384])
    p.add_argument("--hidden_blocks", type=int, nargs="+", default=[3, 3, 3])
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--activation", type=str, default="SiLU")
    # Obs + guidance
    p.add_argument("--obs_type", type=str,
                   choices=["grid", "random"], default="random")
    p.add_argument("--obs_stride", type=int, default=4)
    p.add_argument("--obs_pct", type=float, default=0.05)
    p.add_argument("--obs_fn", type=str, default="linear")
    p.add_argument("--obs_sigma", type=float, default=0.3)
    p.add_argument("--init_sigma", type=float, default=1.0)
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
    # explicit checkpoint
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output")
    # Logging
    p.add_argument("--wandb_project", type=str, default="ScoreDA_SQG")
    p.add_argument("--wandb_entity", type=str, default="stima")
    # match assimilate.py behavior
    p.add_argument("--plot_every", type=int, default=20)
    # Flags
    p.add_argument("--initial_condition", action="store_true",
                   help="Add initial condition.")
    p.add_argument('--data_index', type=int, default=0,
                   help='Index of the data sample to use for assimilation')
    p.add_argument('--start_time', type=int, default=0,
                   help='Start time index for assimilation (default: 0)')
    p.add_argument('--experiment', type=str, default=None,
                   help='Experiment name')
    p.add_argument('--n_times', type=int, default=100,
                   help='Number of time steps for assimilation')
    return p.parse_args()


EXPERIMENTS = {
    'A2': {
        'obs_fn': 'linear',
        'n_ens': 5,
        'obs_pct': 0.05,
        'obs_sigma': 1.0,
        'init_sigma': 1.0,
        'fixed_obs': True,
        'n_times': 100,
        'start_time': 90,
    },
    'A3': {
        'obs_fn': 'arctan',
        'n_ens': 20,
        'obs_pct': 0.25,
        'obs_sigma': 0.01,
        'init_sigma': 1000,
        'fixed_obs': True,
        'n_times': 100,
        'start_time': 5,
    },
    'A4': {
        'obs_fn': 'square_scaled',
        'n_ens': 20,
        'obs_pct': 0.25,
        'obs_sigma': 1.0,
        'init_sigma': 1000,
        'fixed_obs': True,
        'n_times': 100,
        'start_time': 5,
    },
    'A5': {
        'obs_fn': 'linear',
        'n_ens': 20,
        'obs_pct': 0.25,
        'obs_sigma': 5.0,
        'init_sigma': 1000,
        'fixed_obs': True,
        'n_times': 100,
        'start_time': 5,
    },
}


def main():
    args = parse_args()
    if args.experiment is not None:
        assert args.experiment in EXPERIMENTS, f"Experiment '{args.experiment}' not recognized. Available experiments: {list(EXPERIMENTS.keys())}"
        exp_config = EXPERIMENTS[args.experiment]
        for key, value in exp_config.items():
            setattr(args, key, value)
        print(
            f"Using experiment configuration for '{args.experiment}': {exp_config}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.RandomState(42)

    pv_mean = torch.as_tensor(0.0, device=device, dtype=torch.float32)
    pv_std = torch.as_tensor(2672.232, device=device, dtype=torch.float32)

    obs_fn = OBS_FNS[args.obs_fn]

    # ---- Data ----
    DATA_PATH_64 = "/proj/berzelius-2022-164/weather/SQG/test_final/"
    nc_files_64 = [
        f'{DATA_PATH_64}{f[:-4]}' for f in os.listdir(DATA_PATH_64)
        if f.endswith('.npy') and 'sqg_N64_3hrly_steps_110' in f
    ]
    nc_truth = NetCDFDataset(f'{nc_files_64[args.data_index]}.nc', 'r')
    pv_truth_nc = nc_truth.variables['pv']  # (T, 2, ny, nx)
    T, Z, ny, nx = pv_truth_nc.shape
    pv_truth = torch.tensor(
        np.array(pv_truth_nc[args.start_time:T, ...]), dtype=torch.float32, device=device)
    scalefact = nc_truth.f * nc_truth.theta0 / nc_truth.g
    nc_truth.close()

    # NOTE: init_sigma is in unscaled units
    args.init_sigma = args.init_sigma * scalefact

    # ---- Mask ----
    def make_obs_mask(ny_, nx_):
        if args.obs_type == "grid":
            stride = args.obs_stride
            mask = torch.zeros(ny_, nx_, device=device)
            mask[::stride, ::stride] = 1.0
        else:
            nobs = int(ny_ * nx_ * args.obs_pct)
            idx = rng.choice(ny_ * nx_, nobs, replace=False)
            mask = torch.zeros(ny_ * nx_, device=device)
            mask[torch.from_numpy(idx).to(device)] = 1.0
            mask = mask.view(ny_, nx_)
        return mask

    obs_mask = make_obs_mask(ny, nx)                       # (ny, nx)
    iy, ix = (obs_mask > 0.5).nonzero(
        as_tuple=True)    # indices of observed points

    # ---- Checkpoint ----
    payload = torch.load(args.ckpt_path, map_location=device)

    # ---- Build model (same as notebook) ----
    CONFIG = dict(
        window=args.window,
        embedding=args.embedding,
        hidden_channels=tuple(args.hidden_channels),
        hidden_blocks=tuple(args.hidden_blocks),
        kernel_size=args.kernel_size,
        activation=args.activation,
        epochs=4096, batch_size=32, learning_rate=2e-4, weight_decay=1e-3
    )
    score_obj = make_score(**CONFIG).to(device).eval()
    score_obj.load_state_dict(payload["score_state"])

    # ---- A(x): PHYSICAL -> OBS (mask applied framewise) ----
    if args.initial_condition:
        def A_model(x):
            # expects x: (B,L,C,H,W) in temperature units (scalefact × PV)
            B, L, C, H, W = x.shape
            ic_vec = x[:, 0].flatten(start_dim=1)
            x_obs = x[:, 1:, :, iy, ix]
            x_obs_vec = x_obs.reshape(B, -1)
            obs_part = obs_fn(x_obs_vec)

            return torch.cat([ic_vec, obs_part], dim=1)

        x_phys_full = (scalefact * pv_truth).unsqueeze(0)      # (1,T,2,H,W)
        y_clean_full = A_model(x_phys_full)                    # (1, M_full)
        obs_sigma_full = args.obs_sigma * torch.ones_like(y_clean_full)

        # Special (larger) noise for the IC portion (first N entries)
        N = len(pv_truth[0, 0].flatten())
        obs_sigma_full[:, :N] = args.init_sigma  # **2

        # fixed noise reused by prefixing
        eps_full = torch.randn_like(y_clean_full)
        y_star = y_clean_full + obs_sigma_full * eps_full  # (1, M_full)

    else:
        def A_model(x):
            # expects x: (B,L,C,H,W) in temperature units (scalefact × PV)
            B, L, C, H, W = x.shape
            x_obs = x[:, 1:, :, iy, ix]
            x_obs_vec = x_obs.reshape(B, -1)
            return obs_fn(x_obs_vec)

        y_star = A_model(scalefact * pv_truth[None])
        y_star += torch.randn_like(y_star) * args.obs_sigma  # (T,2,H,W)

    # ---- Guided SDE (identical wiring) ----
    scale = (scalefact * pv_std).to(device)
    def score_scaled(x, t, c): return score_obj(x / scale, t, c)

    if args.guidance_method == 'DPS':
        guided_eps = DPSGaussianScore(
            y_star, A=A_model, std=args.obs_sigma,
            sde=ScaledVPSDE(score_scaled,
                            shape=(),
                            scale=scale,
                            ),
            gamma=args.gamma,
            scale=False,
            guidance_strength=args.guidance_strength,
        )
    elif args.guidance_method == 'DPS_scale':
        guided_eps = DPSGaussianScore(
            y_star, A=A_model, std=args.obs_sigma,
            sde=ScaledVPSDE(score_scaled,
                            shape=(),
                            scale=scale,
                            ),
            gamma=args.gamma,
            scale=True,
            guidance_strength=args.guidance_strength,
        )
    elif args.guidance_method == 'MMPS':
        guided_eps = MMPSGaussianScore(
            y_star,
            observation_fn=A_model,
            std=args.obs_sigma,
            init_std=args.init_sigma,
            sde=ScaledVPSDE(score_scaled,
                            shape=(),
                            scale=scale,
                            ),
            guidance_strength=args.guidance_strength,
        )

    guided_sde = ScaledVPSDE(
        eps=guided_eps,
        scale=scale,
        shape=(T, Z, ny, nx),
    ).to(device)

    # ---- Sampling ----
    posterior_list = []
    for e in range(args.n_ens):
        print(f"Sampling ensemble member {e+1}/{args.n_ens}...")
        sample = guided_sde.sample(
            (1,), steps=args.steps, corrections=args.corrections, tau=args.tau)
        posterior_list.append(sample.squeeze(0).detach().cpu())  # (T,2,H,W)
        if device == "cuda":
            torch.cuda.empty_cache()

    posterior_time_first = torch.stack(
        posterior_list, dim=1).numpy()  # (T, ens, 2, H, W)

    # ---- Save NetCDF (optional) ----
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    random_id = str(uuid.uuid4())[:4]  # Use first 4 characters of UUID
    timestamp = datetime.now().strftime("%m_%d_%H")
    file_name = f"SDAsmoothing_{args.experiment}_run_{args.data_index}_{timestamp}_{random_id}"
    nc_filename = Path(args.output_dir) / f"{file_name}.nc"
    nc = NetCDFDataset(nc_filename, mode="w", format="NETCDF4_CLASSIC")
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)
    nc.createDimension("z", 2)
    nc.createDimension("t", T-args.start_time)
    nc.createDimension("ens", args.n_ens)
    ground_truth = nc.createVariable(
        "ground_truth", np.float32, ("t", "z", "y", "x"), zlib=True)
    x_assim = nc.createVariable(
        "x_assim",      np.float32, ("t", "ens", "z", "y", "x"), zlib=True)
    ground_truth[:] = pv_truth.cpu().numpy()
    x_assim[:] = posterior_time_first
    nc.sync()
    nc.close()

    # ---- Metrics (PV units): divide by scalefact to convert from physical ----
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=f"SQG_SDA_smoothing_{args.obs_fn}_{args.obs_pct}pct_{args.guidance_method}"
               f"_strength{args.guidance_strength}_em_steps{args.steps}_corr{args.corrections}_data{args.data_index}",
               config=vars(args))

    for t in range(T-args.start_time):
        # (ens,2,H,W) in scalefact × PV
        pred_t = torch.tensor(posterior_time_first[t]).float()
        gt_t = (scalefact * pv_truth[t].detach().cpu().float())  # (2,H,W)

        wandb.log({
            "RMSE": rmse(torch.mean(pred_t, dim=0), gt_t).item(),
            "CRPS": crps_ens(pred_t, gt_t, ens_dim=0).item(),
            "Spread Skill Ratio": spread_skill_ratio(pred_t, gt_t, ens_dim=0).item(),
        }, step=t+args.start_time)

        # Save spectrum exactly like assimilate.py and log to WandB media
        if t % args.plot_every == 0:
            fname = save_spectrum(pred_t, gt_t.unsqueeze(0), time=t)
            wandb.log({"spectrum": wandb.Image(fname)}, step=t+args.start_time)

    vid_path = save_video(
        torch.tensor(posterior_time_first).float(
        ).cpu(),             # (T,ens,2,H,W)
        (scalefact * pv_truth.clone().detach().float()).cpu(),       # (T,2,H,W)
        args.obs_sigma, args.obs_pct, obs_fn=obs_fn, level=0
    )
    wandb.log({"animation": wandb.Video(vid_path, fps=10, format="mp4")})
    wandb.finish()


if __name__ == "__main__":
    main()
