"""Lazy sliding-window datasets for the processed SWaT/SMD .npz files.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def _read(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

class NormalWindowDataset(Dataset):
    """Windows from a normal-only training file, split chronologically.

    The first (1-val_ratio) portion is train; the final val_ratio is validation.
    Validation windows do not reach back into the training segment.
    """
    def __init__(self, npz_path, window_size=100, stride=1,
                 split="train", val_ratio=0.1, dtype=torch.float32):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if window_size < 2 or stride < 1:
            raise ValueError("window_size must be >=2 and stride >=1")
        if not 0 < val_ratio < 0.5:
            raise ValueError("val_ratio must be between 0 and 0.5")
        data = _read(npz_path)
        X = np.asarray(data["X"], dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"{npz_path} contains NaN/Inf in X; clean before training.")
        self.X = X
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.dtype = dtype
        split_idx = int(len(X) * (1 - val_ratio))
        if split == "train":
            lo, hi = 0, split_idx
        else:
            lo, hi = split_idx, len(X)
        # Valid window start indices must satisfy start + window_size <= hi.
        last_start = hi - self.window_size
        self.starts = np.arange(lo, last_start + 1, self.stride, dtype=np.int64) if last_start >= lo else np.empty(0, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError(f"Not enough rows for {split} windows: segment length={hi-lo}, window={window_size}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        window = self.X[start:start + self.window_size]
        return torch.as_tensor(window, dtype=self.dtype)

class TestWindowDataset(Dataset):
    """Windows from a labelled test file.

    label_mode='last': label is the label of the final timestep.
    label_mode='any': label is 1 if any timestep in the window is anomalous.
    Returns (window_tensor, label_tensor, start_index).
    """
    def __init__(self, npz_path, window_size=100, stride=1,
                 label_mode="last", dtype=torch.float32):
        if label_mode not in {"last", "any"}:
            raise ValueError("label_mode must be 'last' or 'any'")
        if window_size < 2 or stride < 1:
            raise ValueError("window_size must be >=2 and stride >=1")
        data = _read(npz_path)
        self.X = np.asarray(data["X"], dtype=np.float32)
        if "y" not in data:
            raise ValueError(f"{npz_path} has no 'y' labels")
        self.y = np.asarray(data["y"]).reshape(-1).astype(np.int64)
        if len(self.y) != len(self.X):
            raise ValueError(f"X rows ({len(self.X)}) != y labels ({len(self.y)})")
        if not np.isfinite(self.X).all():
            raise ValueError(f"{npz_path} contains NaN/Inf in X; clean before evaluation.")
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.label_mode = label_mode
        self.dtype = dtype
        self.starts = np.arange(0, len(self.X) - self.window_size + 1,
                                self.stride, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError(f"Not enough rows ({len(self.X)}) for window_size={window_size}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + self.window_size
        window = self.X[start:end]
        if self.label_mode == "last":
            label = self.y[end - 1]
        else:
            label = int(np.any(self.y[start:end] == 1))
        return (torch.as_tensor(window, dtype=self.dtype),
                torch.tensor(label, dtype=torch.long),
                start)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train_normal.npz")
    ap.add_argument("--test", required=True, help="Path to test.npz")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    tr = NormalWindowDataset(args.train, args.window, args.stride, "train")
    va = NormalWindowDataset(args.train, args.window, args.stride, "val")
    te = TestWindowDataset(args.test, args.window, args.stride, "last")
    print(f"train windows: {len(tr):,}; one shape: {tuple(tr[0].shape)}")
    print(f"validation windows: {len(va):,}; one shape: {tuple(va[0].shape)}")
    print(f"test windows: {len(te):,}; one shape: {tuple(te[0][0].shape)}")
