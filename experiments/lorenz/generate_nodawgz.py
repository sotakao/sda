# sda/experiments/lorenz/generate.py
from pathlib import Path
import numpy as np
import h5py
from utils import make_chain

# FIXED: point to the directory that contains THIS file, not a Dawgz temp dir
PATH = Path(__file__).resolve().parent  # .../sda/experiments/lorenz

def main():
    chain = make_chain()

    x = chain.prior((1024,))
    x = chain.trajectory(x, length=1024, last=True)
    x = chain.trajectory(x, length=1024)
    x = chain.preprocess(x)
    x = x.transpose(0, 1)

    i = int(0.8 * len(x)); j = int(0.9 * len(x))
    splits = {"train": x[:i], "valid": x[i:j], "test": x[j:]}

    out_dir = PATH / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in splits.items():
        with h5py.File(out_dir / f"{name}.h5", "w") as f:
            f.create_dataset("x", data=arr, dtype=np.float32)

    print("Wrote:", [p.name for p in sorted(out_dir.glob("*.h5"))])

if __name__ == "__main__":
    main()
