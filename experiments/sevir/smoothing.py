#!/usr/bin/env python
import os
import wandb
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from netCDF4 import Dataset as NetCDFDataset

# Your modules
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
    p = argparse.ArgumentParser("Run inference for SDA (parity with notebook).")
    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--hrly_freq", type=int, default=3)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    # Obs + guidance
    p.add_argument("--in_len", type=int, default=6)
    p.add_argument("--obs_pct", type=float, default=0.1)
    p.add_argument("--obs_sigma", type=float, default=0.001)
    p.add_argument("--init_sigma", type=float, default=0.001)
    p.add_argument("--fixed_obs", action="store_true")
    p.add_argument("--n_ens", type=int, default=20)
    p.add_argument("--guidance_method", type=str, choices=["DPS", "DPS_scale", "MMPS"], default="DPS")
    p.add_argument("--guidance_strength", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--corrections", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.5)
    # Checkpoint + output
    p.add_argument("--ckpt_path", type=str, required=True)  # explicit checkpoint
    p.add_argument("--output_dir", type=str, default="./output")
    # Logging
    p.add_argument("--wandb_project", type=str, default="ScoreDA_SQG")
    p.add_argument("--wandb_entity", type=str, default="stima")
    p.add_argument("--plot_every", type=int, default=20)  # match assimilate.py behavior
    # Flags
    p.add_argument("--debug_parity", action="store_true", help="Print first-step invariants and exit.")
    p.add_argument("--initial_condition", action="store_true", help="Add initial condition.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)

    # reproducibility
    torch.manual_seed(0); np.random.seed(0)

    dm = SEVIRLightningDataModule(
            seq_len=25,
            sample_mode='sequent',
            stride=6,
            batch_size=50,
            layout='NTCHW',
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            verbose=False,
            aug_mode='2',
            ret_contiguous=False,
            # datamodule_only
            dataset_name='sevirlr',
            start_date=None,
            train_test_split_date=[2019, 6, 1],
            end_date=None,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            sevir_dir = args.data_dir,
             )
    dm.setup()

    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()

    val_data = val_loader.dataset[3]
    val_data = torch.tensor(val_data, device=device)
    T, C, H, W = val_data.shape

    # Create observation mask and corresponding indices
    mask = (torch.rand(H, W, device=device) < args.obs_pct).float()
    iy, ix = (mask > 0.5).nonzero(as_tuple=True)

    # Condition on first in_len windows + sparse obs
    def A(x): # Masking + initial condition
        # x has shape (B, T, C, H, W)
        B, L, C, H, W = x.shape
        return torch.concatenate([x[:, :args.in_len].flatten(start_dim=1), # Identity map on t=0:5 to impose initial condition
                                  x[:, args.in_len:, :, iy, ix].reshape(B,-1) # Sparse observations for t > 5
                                 ], dim=1)

    # Get noisy observation
    y = A(val_data[None])
    obs_sigma_full = args.obs_sigma * torch.ones_like(y)
    N = H*W*args.in_len
    obs_sigma_full[:, :N] = args.init_sigma**2
    eps_full = torch.randn_like(y)              # fixed noise reused by prefixing
    y = y + obs_sigma_full * eps_full

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
        shape=(T, C, H, W),
    ).to(device)

    # Generate samples
    posterior_list = []
    for e in range(args.n_ens):
        print(f"Sampling ensemble member {e+1}/{args.n_ens}...")
        sample = guided_sde.sample((1,), steps=args.steps, corrections=args.corrections, tau=args.tau)
        posterior_list.append(sample.squeeze(0).detach().cpu())  # (T,C,H,W)
        if device == "cuda":
            torch.cuda.empty_cache()

    posterior_time_first = torch.stack(posterior_list, dim=1).numpy()  # (T, ens, C, H, W)

    # Save NetCDF
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    nc_filename = Path(args.output_dir) / "inference_results.nc"
    nc = NetCDFDataset(nc_filename, mode="w", format="NETCDF4_CLASSIC")
    nc.createDimension("x", W)
    nc.createDimension("y", H)
    nc.createDimension("z", C)
    nc.createDimension("t", T)
    nc.createDimension("ens", args.n_ens)
    ground_truth = nc.createVariable("ground_truth", np.float32, ("t", "z", "y", "x"), zlib=True)
    x_assim      = nc.createVariable("x_assim",      np.float32, ("t", "ens", "z", "y", "x"), zlib=True)
    ground_truth[:] = val_data.cpu().numpy()
    x_assim[:]      = posterior_time_first
    nc.sync(); nc.close()

    # Log metrics and visuals
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                name=f"SEVIR_SDA_smoothing_{args.obs_pct}pct_{args.guidance_method}"
                    f"_strength{args.guidance_strength}_em_steps{args.steps}_corr{args.corrections}",
                config=vars(args))

    for t in range(T):
        pred_t  = torch.tensor(posterior_time_first[t]).float()   # (ens,C,H,W)
        gt_t    = (val_data[t].detach().cpu().float())  # (C,H,W)

        wandb.log({
            "RMSE": rmse(torch.mean(pred_t, dim=0), gt_t).item(),
            "CRPS": crps_ens(pred_t, gt_t, ens_dim=0).item(),
            "Spread Skill Ratio": spread_skill_ratio(pred_t, gt_t, ens_dim=0).item(),
        }, step=t)

        # # Save spectrum exactly like assimilate.py and log to WandB media
        # if t % args.plot_every == 0:
        #     fname = save_spectrum(pred_t, gt_t.unsqueeze(0), time=t)
        #     wandb.log({"spectrum": wandb.Image(fname)}, step=t)

    # (optional) visuals
    vid_path = save_video(
        torch.tensor(posterior_time_first).float().cpu(),             # (T,ens,2,H,W)
        (val_data.clone().detach().float()).cpu(),       # (T,C,H,W)
        args.obs_sigma, args.obs_pct, obs_fn=lambda x: x, level=0
    )
    wandb.log({"animation": wandb.Video(vid_path, fps=10, format="mp4")})
    wandb.finish()


if __name__ == "__main__":
    main()
    

