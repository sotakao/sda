#!/usr/bin/env python
import os
import argparse
from types import SimpleNamespace
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from netCDF4 import Dataset as NetCDFDataset

# Your modules
from sda.score import (
    GaussianScore, ScaledVPSDE,
    DPSGaussianScore, MMPSGaussianScore,
    MCScoreNet, ScoreUNet
)
from utils import OBS_FNS, ACTIVATIONS, save_spectrum, save_video
from metrics import rmse, crps_ens, spread_skill_ratio


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
    p = argparse.ArgumentParser("Run inference for SDA (parity with notebook).")
    # Data
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_file", type=str, default="sqg_pv_train.h5")
    p.add_argument("--hrly_freq", type=int, default=3)
    # Model
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--embedding", type=int, default=64)
    p.add_argument("--hidden_channels", type=int, nargs="+", default=[96, 192, 384])
    p.add_argument("--hidden_blocks", type=int, nargs="+", default=[3, 3, 3])
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--activation", type=str, default="SiLU")
    # Obs + guidance
    p.add_argument("--obs_type", type=str, choices=["grid", "random"], default="random")
    p.add_argument("--obs_stride", type=int, default=4)
    p.add_argument("--obs_pct", type=float, default=0.05)
    p.add_argument("--obs_fn", type=str, default="linear")
    p.add_argument("--obs_sigma", type=float, default=0.3)
    p.add_argument("--fixed_obs", action="store_true")
    p.add_argument("--n_ens", type=int, default=20)
    p.add_argument("--guidance_method", type=str, choices=["DPS", "DPS_scale", "MMPS"], default="DPS")
    p.add_argument("--guidance_strength", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--corrections", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.5)
    # Checkpoint + output
    p.add_argument("--ckpt_path", type=str, default="")  # explicit checkpoint
    p.add_argument("--ckpt_dir", type=str, default="../../runs_sqg")  # fallback root
    p.add_argument("--output_dir", type=str, default="./output")
    # Logging
    p.add_argument("--log_wandb", type=int, default=1)
    p.add_argument("--wandb_project", type=str, default="ScoreDA_SQG")
    p.add_argument("--wandb_entity", type=str, default="stima")
    p.add_argument("--plot_every", type=int, default=20)  # match assimilate.py behavior
    # Debug
    p.add_argument("--debug_parity", action="store_true", help="Print first-step invariants and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)

    # (Optional) reproducibility
    torch.manual_seed(0); np.random.seed(0)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # ---- Defaults matching your notebook ----
    pv_mean = torch.as_tensor(0.0, device=device, dtype=torch.float32)
    pv_std  = torch.as_tensor(2672.232, device=device, dtype=torch.float32)

    obs_fn = OBS_FNS[args.obs_fn]
    rng = np.random.RandomState(42)

    # ---- Data ----
    # test_file = Path(args.data_dir) / f"{args.hrly_freq}hrly" / f"sqg_N64_{args.hrly_freq}hrly_100.nc"
    test_file = f"/resnick/groups/astuart/sotakao/score-based-ensemble-filter/EnSFInpainting/data/test/sqg_N64_{args.hrly_freq}hrly_100.nc"
    nc_truth = NetCDFDataset(test_file, 'r')
    pv_truth_nc = nc_truth.variables['pv']  # (T, 2, ny, nx)
    T, Z, ny, nx = pv_truth_nc.shape
    pv_truth = torch.tensor(np.array(pv_truth_nc[:T, ...]), dtype=torch.float32, device=device)
    scalefact = nc_truth.f * nc_truth.theta0 / nc_truth.g
    nc_truth.close()

    # ---- Mask (same logic) ----
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
    mask4d = obs_mask.view(1, 1, ny, nx)                   # broadcast over (B,L,C)

    # ---- Checkpoint ----
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_dir = Path(args.ckpt_dir) / f"mcscore_vpsde_sqg_window_{args.window}" / "checkpoints"
        files = sorted(ckpt_dir.glob("epoch_*.pt"))
        if not files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}.")
        ckpt_path = files[-1]
    payload = torch.load(ckpt_path, map_location=device)

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
    def A_model(x):
        # x: (B,L,C,H,W) in physical units (scalefact × PV)
        return obs_fn(x) * mask4d

    # ---- Observations in OBS units (masked) ----
    noise  = torch.randn_like(pv_truth) * args.obs_sigma
    y_star = obs_fn(scalefact * pv_truth) * obs_mask + noise * obs_mask  # (T,2,H,W)

    # ---- Guided SDE (identical wiring) ----
    scale = (scalefact * pv_std).to(device)
    score_scaled = lambda x, t, c: score_obj(x / scale, t, c)

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
            sde=ScaledVPSDE(score_scaled,
                            shape=(),
                            scale=scale,
            ),
        )
    guided_sde = ScaledVPSDE(
        eps=guided_eps,
        scale=scale,
        shape=y_star.shape,
    ).to(device)

    # ---- Parity probe (prints invariants) ----
    if args.debug_parity:
        with torch.no_grad():
            B = 1
            x0 = (torch.randn(B, T, Z, ny, nx, device=device) * scale)
            time = torch.linspace(1, 0, args.steps + 1, device=device)
            t0, dt = time[0], 1.0 / args.steps
            r    = guided_sde.mu(t0 - dt) / guided_sde.mu(t0)
            eps0 = guided_sde.eps(x0, t0, c=None)
            x1   = r * x0 + (guided_sde.sigma(t0 - dt) - r * guided_sde.sigma(t0)) * eps0

            print("DEBUG invariants:")
            print("  scale:", float(scale))
            print("  mu0, sigma0:", float(guided_sde.mu(t0)), float(guided_sde.sigma(t0)))
            print("  ||eps0||, ||x1 - r*x0||:", float(eps0.norm()), float((x1 - r*x0).norm()))
            print("  A(x0) mean:", float(A_model(x0).abs().mean()))
            print("  y mean:", float(y_star.abs().mean()))
        return

    # ---- Sampling ----
    posterior_list = []
    for e in range(args.n_ens):
        print(f"Sampling ensemble member {e+1}/{args.n_ens}...")
        sample = guided_sde.sample((1,), steps=args.steps, corrections=args.corrections, tau=args.tau)
        posterior_list.append(sample.squeeze(0).detach().cpu())  # (T,2,H,W)
        if device == "cuda":
            torch.cuda.empty_cache()

    posterior_time_first = torch.stack(posterior_list, dim=1).numpy()  # (T, ens, 2, H, W)

    # ---- Save NetCDF (optional) ----
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    nc_filename = Path(args.output_dir) / "inference_results.nc"
    nc = NetCDFDataset(nc_filename, mode="w", format="NETCDF4_CLASSIC")
    nc.createDimension("x", nx)
    nc.createDimension("y", ny)
    nc.createDimension("z", 2)
    nc.createDimension("t", T)
    nc.createDimension("ens", args.n_ens)
    ground_truth = nc.createVariable("ground_truth", np.float32, ("t", "z", "y", "x"), zlib=True)
    x_assim      = nc.createVariable("x_assim",      np.float32, ("t", "ens", "z", "y", "x"), zlib=True)
    ground_truth[:] = pv_truth.cpu().numpy()
    x_assim[:]      = posterior_time_first
    nc.sync(); nc.close()

    # ---- Metrics (PV units): divide by scalefact to convert from physical ----
    if args.log_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f"SDA_{args.obs_fn}_{args.obs_pct}pct_{args.guidance_method}"
                        f"_strength{args.guidance_strength}_em_steps{args.steps}_corr{args.corrections}",
                   config=vars(args))

        for t in range(T):
            pred_t  = torch.tensor(posterior_time_first[t]).float()   # (ens,2,H,W) in scalefact × PV
            gt_t    = (scalefact * pv_truth[t].detach().cpu().float())  # (2,H,W)

            wandb.log({
                "RMSE": rmse(torch.mean(pred_t, dim=0), gt_t).item(),
                "CRPS": crps_ens(pred_t, gt_t, ens_dim=0).item(),
                "Spread Skill Ratio": spread_skill_ratio(pred_t, gt_t, ens_dim=0).item(),
            }, step=t)

            # Save spectrum exactly like assimilate.py and log to WandB media
            if t % args.plot_every == 0:
                fname = save_spectrum(pred_t, gt_t.unsqueeze(0), time=t)
                wandb.log({"spectrum": wandb.Image(fname)}, step=t)

        # (optional) visuals
        vid_path = save_video(
            torch.tensor(posterior_time_first).float().cpu(),             # (T,ens,2,H,W)
            (scalefact * pv_truth.clone().detach().float()).cpu(),       # (T,2,H,W)
            args.obs_sigma, args.obs_pct, obs_fn=obs_fn, level=0
        )
        wandb.log({"animation": wandb.Video(vid_path, fps=10, format="mp4")})
        wandb.finish()
    else:
        pred_pv = torch.tensor(posterior_time_first).float() / float(scalefact)  # (T, ens, 2, H, W)
        ens_mean = pred_pv.mean(dim=1)                                           # (T, 2, H, W)
        gt       = pv_truth.detach().cpu().float()                               # (T, 2, H, W)

        # If rmse() returns per-time values (e.g., shape (T,) or (T,2)), average them:
        rmse_t   = rmse(ens_mean, gt)            # tensor, not a scalar
        rmse_all = rmse_t.mean().item()          # make it a scalar

        print(f"RMSE(all): {rmse_all:.4f}")


if __name__ == "__main__":
    main()
