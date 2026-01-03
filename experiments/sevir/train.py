import os
import math
import argparse
import torch
import dotenv
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from pathlib import Path
from typing import *
from torch.utils.data import Dataset

from sda.utils import loop
from sda.score import MCScoreNet, ScoreUNet, VPSDE
from utils import ACTIVATIONS, SEVIRLightningDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dotenv.load_dotenv()

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# ---------------------------
# Dataset wrapper: SEVIR -> match TrajectoryDataset behavior
# ---------------------------
class SEVIRWindowDataset(Dataset):
    """
    Wraps dm.sevir_train / dm.sevir_val (torch.utils.data.Subset of SEVIRTorchDataset).
    SEVIRTorchDataset returns shape determined by layout:
      with layout="NTHWC" -> datamodule passes "THWC" -> dataset replaces C->1 -> "THW1"
      so sample is (T, H, W, C) with C usually 1.

    This wrapper does the same operations as TrajectoryDataset:
      - optional random time window crop
      - optional flatten time+channel -> (window*C, H, W)
    """
    def __init__(self, base: Dataset, window: Optional[int], flatten: bool = True):
        super().__init__()
        self.base = base
        self.window = window
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = self.base[i]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()

        # Expect (T,H,W,C) (for sevirlr typically C=1)
        if x.ndim == 3:
            # (T,H,W) -> (T,H,W,1)
            x = x.unsqueeze(-1)
        if x.ndim != 4:
            raise ValueError(f"Expected SEVIR sample of shape (T,H,W,C), got {tuple(x.shape)}")

        T, H, W, C = x.shape

        if self.window is not None:
            w = min(self.window, T)
            start_max = max(T - w, 0)
            start = 0 if start_max == 0 else torch.randint(0, start_max + 1, size=()).item()
            x = x[start:start + w]  # (w,H,W,C)

        # (w,H,W,C) -> (w,C,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()

        if self.flatten:
            # (w,C,H,W) -> (w*C,H,W)
            x = x.flatten(0, 1)

        return x, {}


# ---------------------------
# Model factory
# ---------------------------
class LocalScoreUNet(ScoreUNet):
    """Creates a score U-Net with a forcing channel (same as your SQG script)."""

    def __init__(
        self,
        channels: int,
        size: int,
        periodic: bool = False,
        **kwargs,
    ):
        super().__init__(channels, 1, periodic=periodic, **kwargs)

        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()
        self.register_buffer("forcing", forcing)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.forcing)


def make_score(
    window: int,
    data_channels: int,
    embedding: int,
    hidden_channels: Tuple[int, ...],
    hidden_blocks: Tuple[int, ...],
    kernel_size: int,
    activation: str,
    spatial_size: int,
    periodic: bool = False,
    **absorb
) -> torch.nn.Module:
    score = MCScoreNet(2, order=window // 2)
    score.kernel = LocalScoreUNet(
        channels=window * data_channels,  # <-- CHANGED from window*2
        size=spatial_size,               # <-- CHANGED from fixed 64
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        periodic=periodic,
        padding_mode="circular",
    )
    return score


def flush_history(train_hist, valid_hist, lr_hist, hist_dir, tag="latest"):
    np.savez(
        hist_dir / f"history_{tag}.npz",
        train=np.asarray(train_hist, dtype=np.float64),
        valid=np.asarray(valid_hist, dtype=np.float64),
        lr=np.asarray(lr_hist, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser(description="Train SEVIR model with configurable parameters.")

    # ---- SEVIR-specific ----
    parser.add_argument("--sevir_dir", type=str, default="/resnick/groups/astuart/sotakao/score-based-ensemble-filter/FlowDAS/experiments/weather_forecasting/data/sevir_lr", help="Root directory containing SEVIR data (CATALOG.csv, data/).")
    parser.add_argument("--dataset_name", type=str, default="sevirlr", choices=["sevir", "sevirlr"])
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--sample_mode", type=str, default="sequent")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--layout", type=str, default="NTHWC", help="Loader layout; dataset receives layout without N.")
    parser.add_argument("--train_test_split_date", type=int, nargs=3, default=[2019, 6, 1])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--aug_mode", type=str, default="2", choices=["0", "1", "2"])
    parser.add_argument("--num_workers", type=int, default=4)

    # ---- model/training ----
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[96, 192, 384])
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[3, 3, 3])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--epochs", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size used by sda.loop()")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    # ---- wandb ----
    parser.add_argument("--wandb_project", type=str, default="sevir_training")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    # Output dirs
    out_root = Path("./runs_sevir")
    out_root.mkdir(parents=True, exist_ok=True)

    exp_name = (
        f"{args.dataset_name}_window{args.window}_epochs{args.epochs}_{'periodic' if args.periodic else ' '}"
    )
    run_dir = out_root / exp_name
    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    hist_dir = run_dir / "history"
    for d in [run_dir, ckpt_dir, fig_dir, hist_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # WandB init
    if WANDB:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", args.wandb_project),
            entity=os.getenv("WANDB_ENTITY", args.wandb_entity),
            name=exp_name,
            config=vars(args),
        )

    # Build SEVIR datamodule + datasets
    dm = SEVIRLightningDataModule(
        seq_len=args.seq_len,
        sample_mode=args.sample_mode,
        stride=args.stride,
        batch_size=1,                 # dataset is item-based; loop() will create its own loader
        layout=args.layout,           # e.g. NTHWC
        output_type=np.float32,
        preprocess=True,
        rescale_method="01",
        verbose=False,
        aug_mode=args.aug_mode,
        ret_contiguous=False,
        dataset_name=args.dataset_name,
        start_date=None,
        train_test_split_date=tuple(args.train_test_split_date),
        end_date=None,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        sevir_dir=args.sevir_dir,
    )
    dm.setup()

    trainset = SEVIRWindowDataset(dm.sevir_train, window=args.window, flatten=True)
    validset = SEVIRWindowDataset(dm.sevir_val,   window=args.window, flatten=True)

    # Infer shapes
    x0, _ = trainset[0]  # (window*C, H, W)
    in_channels, H, W = tuple(x0.shape)

    # For SEVIRLR this should usually be 1, but infer from raw sample too:
    raw0 = dm.sevir_train[0]
    if raw0.ndim == 3:
        data_channels = 1
    elif raw0.ndim == 4:
        data_channels = int(raw0.shape[-1])
    else:
        data_channels = 1

    print(f"[data] flattened sample: {tuple(x0.shape)} ; inferred H,W=({H},{W}) ; data_channels={data_channels}")

    # Model config (what loop() expects)
    CONFIG = {
        "window": args.window,
        "embedding": args.embedding,
        "hidden_channels": tuple(args.hidden_channels),
        "hidden_blocks": tuple(args.hidden_blocks),
        "kernel_size": args.kernel_size,
        "activation": args.activation,
        "periodic": args.periodic,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }

    # Build model + SDE
    score = make_score(
        window=args.window,
        data_channels=data_channels,
        embedding=args.embedding,
        hidden_channels=tuple(args.hidden_channels),
        hidden_blocks=tuple(args.hidden_blocks),
        kernel_size=args.kernel_size,
        activation=args.activation,
        spatial_size=H,
        periodic=args.periodic,
    )
    sde = VPSDE(score.kernel, shape=(in_channels, H, W)).to(device)

    # Plot + (optional) wandb image
    def save_samples_figure(
        sde_module: VPSDE,
        epoch: int,
        ts=(0, 3, 6, 9),
        steps: int = 64,
        log_to_wandb: bool = False,
    ):
        sde_module.eval()
        with torch.no_grad():
            x = sde_module.sample(steps=steps).detach().cpu()  # expected (T, C, H, W) for many samplers

        # Make robust to different return shapes
        if x.ndim == 3:
            # (T,H,W) -> pretend single channel
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Unexpected sample shape from sde.sample(): {tuple(x.shape)}")

        T = x.shape[0]
        chosen = [t for t in ts if t < T] or [0]

        n = len(chosen)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

        for k, t in enumerate(chosen):
            r, c = divmod(k, cols)
            im = axes[r, c].imshow(x[t, 0], origin="upper")
            axes[r, c].set_title(f"t = {t}")
            fig.colorbar(im, ax=axes[r, c])

        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")

        fig.suptitle(f"Samples @ epoch {epoch}")
        fig.tight_layout()
        out_path = fig_dir / f"samples_epoch_{epoch:06d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        if log_to_wandb and WANDB:
            wandb.log({"samples": wandb.Image(str(out_path), caption=f"Samples @ epoch {epoch}")})

        return out_path

    # Train
    generator = loop(
        sde,
        trainset,
        validset,
        device=device,
        **CONFIG,
    )

    train_history, valid_history, lr_history = [], [], []

    for epoch, (loss_train, loss_valid, lr) in enumerate(generator, start=1):
        train_history.append(float(loss_train))
        valid_history.append(float(loss_valid))
        lr_history.append(float(lr))

        if WANDB:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": float(loss_train),
                    "valid_loss": float(loss_valid),
                    "learning_rate": float(lr),
                },
                step=epoch,
            )

        if epoch % 100 == 0 or epoch == args.epochs:
            flush_history(train_history, valid_history, lr_history, hist_dir, tag=f"epoch_{epoch:06d}")
            flush_history(train_history, valid_history, lr_history, hist_dir, tag="latest")
            save_samples_figure(sde, epoch=epoch, ts=(0, 3, 6, 9), steps=64, log_to_wandb=True)

            ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch:06d}_{'per' if args.periodic else 'aper'}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": sde.eps.state_dict(),
                    "config": CONFIG,
                    "sevir_args": vars(args),
                    "in_channels": in_channels,
                    "H": H,
                    "W": W,
                    "data_channels": data_channels,
                },
                ckpt_path,
            )
            latest_path = ckpt_dir / f"latest_{'per' if args.periodic else 'aper'}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": sde.eps.state_dict(),
                    "config": CONFIG,
                    "sevir_args": vars(args),
                    "in_channels": in_channels,
                    "H": H,
                    "W": W,
                    "data_channels": data_channels,
                },
                latest_path,
            )

    flush_history(train_history, valid_history, lr_history, hist_dir, tag="final")
    print("✅ Done.")


if __name__ == "__main__":
    main()
