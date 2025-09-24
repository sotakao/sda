import os
import json
import math
import glob
from pathlib import Path
from typing import *
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt

from sda.mcs import *
from sda.score import *
from sda.utils import *

window = 5

# ---------------------------
# Datasets
# ---------------------------

class TrajectoryDataset(Dataset):
    def __init__(
        self,
        file: Union[str, Path],
        window: int = None,
        flatten: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        file = str(file)
        with h5py.File(file, mode='r') as f:
            self.data = f['x'][:]

        self.window = window
        self.flatten = flatten
        self.normalize = normalize

        self.mean = self.data.mean()
        self.std = self.data.std() if self.data.std() > 0 else 1.0

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = torch.from_numpy(self.data[i])

        if self.normalize:
            x = (x - self.mean) / self.std

        if self.window is not None:
            # guard against short trajectories
            T = x.shape[0]
            w = min(self.window, T)
            start_max = max(T - w, 0)
            start = 0 if start_max == 0 else torch.randint(0, start_max + 1, size=()).item()
            x = torch.narrow(x, dim=0, start=start, length=w)

        if self.flatten:
            return x.flatten(0, 1), {}
        else:
            return x, {}

# ---------------------------
# Model factory
# ---------------------------

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
    window: int = 3,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (64, 128, 256),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
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
# Paths & config
# ---------------------------

data_dir   = "/central/scratch/sotakao/sqg_train_data/3hrly"
train_fname = "sqg_pv_train.h5"
valid_fname = "sqg_pv_valid.h5"

out_root = Path("./runs_sqg")      # output root (you can change)
out_root.mkdir(parents=True, exist_ok=True)

exp_name = f"mcscore_vpsde_sqg_window_{window}"
run_dir  = out_root / exp_name
ckpt_dir = run_dir / "checkpoints"
fig_dir  = run_dir / "figures"
hist_dir = run_dir / "history"

for d in [run_dir, ckpt_dir, fig_dir, hist_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Data
# ---------------------------
trainset = TrajectoryDataset(Path(data_dir) / train_fname, normalize=True, window=window, flatten=True)
validset = TrajectoryDataset(Path(data_dir) / valid_fname, normalize=True, window=window, flatten=True)

CONFIG = {
    # Architecture
    'window': window,
    'embedding': 64,
    'hidden_channels': (96, 192, 384),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Training
    'epochs': 4096,
    'batch_size': 32,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'cosine',
}

# ---------------------------
# Build model + SDE
# ---------------------------

score = make_score(**CONFIG)
sde = VPSDE(score.kernel, shape=(CONFIG['window'] * 2, 64, 64)).cuda()

# ---------------------------
# Checkpoint helpers
# ---------------------------

def _ckpt_path(epoch: int) -> Path:
    return ckpt_dir / f"epoch_{epoch:06d}.pt"

def _latest_ckpt() -> Optional[Path]:
    files = sorted(ckpt_dir.glob("epoch_*.pt"))
    return files[-1] if files else None

def save_checkpoint(epoch: int,
                    score: nn.Module,
                    sde: nn.Module,
                    train_hist: List[float],
                    valid_hist: List[float],
                    lr_hist: List[float]) -> Path:
    payload = {
        "epoch": epoch,
        "score_state": score.state_dict(),
        "sde_state": sde.state_dict(),
        "train_history": train_hist,
        "valid_history": valid_hist,
        "lr_history": lr_hist,
        "config": CONFIG,
    }
    p = _ckpt_path(epoch)
    torch.save(payload, p)
    return p

def load_checkpoint(path: Path,
                    score: nn.Module,
                    sde: nn.Module) -> Tuple[int, List[float], List[float], List[float], dict]:
    payload = torch.load(path, map_location="cuda")
    score.load_state_dict(payload["score_state"])
    sde.load_state_dict(payload["sde_state"])
    epoch = int(payload.get("epoch", 0))
    train_hist = list(payload.get("train_history", []))
    valid_hist = list(payload.get("valid_history", []))
    lr_hist = list(payload.get("lr_history", []))
    cfg = payload.get("config", {})
    return epoch, train_hist, valid_hist, lr_hist, cfg

# ---------------------------
# History save helpers
# ---------------------------

def flush_history(train_hist, valid_hist, lr_hist, tag="latest"):
    # .npz
    np.savez(hist_dir / f"history_{tag}.npz",
             train=np.asarray(train_hist, dtype=np.float64),
             valid=np.asarray(valid_hist, dtype=np.float64),
             lr=np.asarray(lr_hist, dtype=np.float64))
    # .json (tiny + human-readable)
    with open(hist_dir / f"history_{tag}.json", "w") as f:
        json.dump({"train": train_hist, "valid": valid_hist, "lr": lr_hist}, f)

# ---------------------------
# Sample plotting
# ---------------------------

def save_samples_figure(sde_module: VPSDE,
                        epoch: int,
                        ts=(0,3,6,9),
                        steps: int = 64):
    sde_module.eval()
    with torch.no_grad():
        # Generates a trajectory of latent samples across time steps (API: sde.sample(steps) -> [T, C, H, W] or [T, ...])
        x = sde_module.sample(steps=steps).detach().cpu()
        x = x.unflatten(0, (-1, 2))

    # make sure requested indices exist
    T = x.shape[0]
    chosen = [t for t in ts if t < T]
    if not chosen:
        chosen = [0]

    n = len(chosen)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)

    for k, t in enumerate(chosen):
        r, c = divmod(k, cols)
        im = axes[r, c].imshow(x[t][0], origin='upper')
        axes[r, c].set_title(f"t = {t}")
        fig.colorbar(im, ax=axes[r, c])

    # hide unused axes
    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Samples @ epoch {epoch}")
    fig.tight_layout()
    out_path = fig_dir / f"samples_epoch_{epoch:06d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# ---------------------------
# Resume logic (model weights + history)
# ---------------------------

start_epoch_global = 0
train_history: List[float] = []
valid_history: List[float] = []
lr_history:    List[float] = []

latest = _latest_ckpt()
if latest is not None:
    print(f"[resume] Found checkpoint: {latest}")
    last_epoch, train_history, valid_history, lr_history, _ = load_checkpoint(latest, score, sde)
    start_epoch_global = last_epoch + 1
    print(f"[resume] Resuming from epoch {start_epoch_global}")
else:
    print("[resume] No checkpoint found. Starting fresh.")

# Adjust planned epochs so we finish at CONFIG['epochs'] total
TOTAL_EPOCHS = CONFIG['epochs']
remaining = max(TOTAL_EPOCHS - start_epoch_global, 0)
if remaining == 0:
    print("[info] Training already completed according to checkpoints.")
else:
    # Create a copy of CONFIG for the loop() to specify remaining epochs
    CONFIG_RUN = dict(CONFIG)
    CONFIG_RUN['epochs'] = remaining

    # ---------------------------
    # Training
    # ---------------------------
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
        **CONFIG_RUN,
    )

    # Iterate, mapping local epoch index -> global epoch index
    # local runs from 0..remaining-1; global = start_epoch_global + local
    for local_idx, (loss_train, loss_valid, lr) in enumerate(generator):
        global_epoch = start_epoch_global + local_idx

        train_history.append(float(loss_train))
        valid_history.append(float(loss_valid))
        lr_history.append(float(lr))

        # checkpoint every 100 epochs (and at the very end)
        need_ckpt = ((global_epoch + 1) % 100 == 0) or (global_epoch + 1 == TOTAL_EPOCHS)
        if need_ckpt:
            # Save model + history
            ckpt_path = save_checkpoint(global_epoch, score, sde, train_history, valid_history, lr_history)
            print(f"[ckpt] Saved: {ckpt_path}")

            # Save a figure of samples at t = 0,3,6,9
            fig_path = save_samples_figure(sde, epoch=global_epoch, ts=(0,3,6,9), steps=64)
            print(f"[fig ] Saved: {fig_path}")

            # Flush history snapshots
            flush_history(train_history, valid_history, lr_history, tag=f"epoch_{global_epoch:06d}")
            flush_history(train_history, valid_history, lr_history, tag="latest")

# Always save final histories (redundant but handy)
flush_history(train_history, valid_history, lr_history, tag="final")

# Optional quick visualization snippet (kept from your original)
# t = 0
# x = sde.sample(steps=100).cpu()
# plt.imshow(x[t][0]); plt.colorbar(); plt.show()
