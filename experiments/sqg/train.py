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
from sda.utils import loop
from sda.score import MCScoreNet, ScoreUNet, VPSDE
from utils import TrajectoryDataset, ACTIVATIONS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load environments for WandB
dotenv.load_dotenv()

# Optional logging
try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


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


def main():
    parser = argparse.ArgumentParser(description="Train SQG model with configurable parameters.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training and validation data.")
    parser.add_argument("--train_file", type=str, default="sqg_pv_train.h5", help="Training dataset filename.")
    parser.add_argument("--valid_file", type=str, default="sqg_pv_valid.h5", help="Validation dataset filename.")
    parser.add_argument("--hrly_freq", type=int, default=3, help="Frequency of hourly data.")
    parser.add_argument("--window", type=int, default=5, help="Window size for the dataset.")
    parser.add_argument("--embedding", type=int, default=64, help="Embedding size for the model.")
    parser.add_argument("--hidden_channels", type=int, nargs="+", default=[96, 192, 384], help="Hidden channels for the model.")
    parser.add_argument("--hidden_blocks", type=int, nargs="+", default=[3, 3, 3], help="Hidden blocks for the model.")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the model.")
    parser.add_argument("--activation", type=str, default="SiLU", help="Activation function for the model.")
    parser.add_argument("--epochs", type=int, default=4096, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for the optimizer.")
    parser.add_argument("--wandb_project", type=str, default="sqg_training", help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name.")
    args = parser.parse_args()

    # Setup directories
    out_root = Path("./runs_sqg")
    out_root.mkdir(parents=True, exist_ok=True)
    exp_name = f"{args.hrly_freq}hrly_window_size_{args.window}_num_epochs_{args.epochs}"
    run_dir = out_root / exp_name
    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    hist_dir = run_dir / "history"
    for d in [run_dir, ckpt_dir, fig_dir, hist_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    if WANDB:
        wandb.init(project=os.getenv("WANDB_PROJECT", args.wandb_project),
                   entity=os.getenv("WANDB_ENTITY", None),
                   name=exp_name,
                   config=vars(args))

    # Load datasets
    trainset = TrajectoryDataset(
        Path(args.data_dir) / f"{args.hrly_freq}hrly" / args.train_file,
        normalize=True,
        window=args.window,
        flatten=True,
    )
    validset = TrajectoryDataset(
        Path(args.data_dir) / f"{args.hrly_freq}hrly" / args.valid_file,
        normalize=True,
        window=args.window,
        flatten=True,
        mean=trainset.mean,
        std=trainset.std,
    )

    # Model configuration
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
    score = make_score(**CONFIG)
    sde = VPSDE(score.kernel, shape=(args.window * 2, 64, 64)).cuda()

    def save_samples_figure(sde_module: VPSDE, epoch: int, ts=(0, 3, 6, 9), steps: int = 64, log_to_wandb: bool = False):
        sde_module.eval()
        with torch.no_grad():
            x = sde_module.sample(steps=steps).detach().cpu()
            x = x.unflatten(0, (-1, 2))

        T = x.shape[0]
        chosen = [t for t in ts if t < T]
        if not chosen:
            chosen = [0]

        n = len(chosen)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

        for k, t in enumerate(chosen):
            r, c = divmod(k, cols)
            im = axes[r, c].imshow(x[t][0], origin='upper')
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

    # Training loop
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

        # Log to WandB
        if WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": loss_train,
                "valid_loss": loss_valid,
                "learning_rate": lr,
            })

        # Save samples and history periodically
        if epoch % 100 == 0 or epoch == args.epochs:
            flush_history(train_history, valid_history, lr_history, hist_dir, tag=f"epoch_{epoch:06d}")
            flush_history(train_history, valid_history, lr_history, hist_dir, tag="latest")
            save_samples_figure(sde, epoch=epoch, ts=(0, 3, 6, 9), steps=64, log_to_wandb=True)

    # Final history save
    flush_history(train_history, valid_history, lr_history, hist_dir, tag="final")

def flush_history(train_hist, valid_hist, lr_hist, hist_dir, tag="latest"):
    np.savez(hist_dir / f"history_{tag}.npz",
             train=np.asarray(train_hist, dtype=np.float64),
             valid=np.asarray(valid_hist, dtype=np.float64),
             lr=np.asarray(lr_hist, dtype=np.float64))


if __name__ == "__main__":
    main()

