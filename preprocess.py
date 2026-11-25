#!/usr/bin/env python3
"""
Preprocess extracted SWaT and SMD folders for one-class anomaly detection.

Expected layout:
  swat_smd_preprocessing/
    preprocess.py
    Datasets/
      SWAT/  (contains normal.csv and merged.csv, possibly nested)
      SMD/   (contains train/, test/, test_label/, possibly nested)

Install:
  python -m pip install numpy pandas scikit-learn

Run from the project folder:
  python preprocess.py --swat-dir Datasets/SWAT --smd-dir Datasets/SMD --out processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities/missing values using interpolation, then median fallback."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", limit_direction="both", axis=0)
    medians = df.median(numeric_only=True)
    return df.fillna(medians).fillna(0.0)


def find_unique_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} anywhere under {root}")
    matches.sort(key=lambda p: (len(p.parts), str(p).lower()))
    if len(matches) > 1:
        print(f"[Warning] Multiple {filename} files found; using {matches[0]}")
    return matches[0]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def select_numeric_features(
    df: pd.DataFrame, excluded: set[str]
) -> tuple[pd.DataFrame, list[str]]:
    converted = {}
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            converted[col] = values
    if not converted:
        raise ValueError("No numeric sensor columns found. Check the input data.")
    result = pd.DataFrame(converted, index=df.index)
    return result, list(converted.keys())


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def preprocess_swat(root: Path, out_root: Path) -> None:
    normal_path = find_unique_file(root, "normal.csv")
    merged_path = find_unique_file(root, "merged.csv")
    print(f"[SWaT] Normal file: {normal_path}")
    print(f"[SWaT] Evaluation file: {merged_path}")

    normal = read_csv(normal_path)
    merged = read_csv(merged_path)

    time_col = next(
        (c for c in normal.columns if c.strip().lower() in {"timestamp", "time"}),
        None,
    )
    if time_col and time_col in merged.columns:
        # SWaT timestamps commonly use day/month/year formatting.
        normal[time_col] = pd.to_datetime(
            normal[time_col], errors="coerce", dayfirst=True
        )
        merged[time_col] = pd.to_datetime(
            merged[time_col], errors="coerce", dayfirst=True
        )

    label_candidates = {"normal/attack", "normal_attack", "label", "attack"}
    label_col = next(
        (c for c in merged.columns if c.strip().lower() in label_candidates), None
    )

    train_excluded = {time_col} if time_col else set()
    train_df, feature_cols = select_numeric_features(normal, train_excluded)
    train_df = clean_numeric(train_df)

    # Fit scaler on training-normal data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        train_df.to_numpy(dtype=np.float32)
    ).astype(np.float32)

    test_excluded = {time_col} if time_col else set()
    if label_col:
        test_excluded.add(label_col)
    test_df, _ = select_numeric_features(merged, test_excluded)

    # Match evaluation feature order to training feature order.
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan
    test_df = clean_numeric(test_df[feature_cols])
    X_test = scaler.transform(
        test_df.to_numpy(dtype=np.float32)
    ).astype(np.float32)

    if label_col:
        labels = merged[label_col].astype(str).str.strip().str.lower()
        y_test = (~labels.isin({"normal", "0", "false", "benign"})).astype(
            np.int8
        ).to_numpy()
    else:
        print("[SWaT] WARNING: no label column found; y is set to -1.")
        y_test = np.full(len(merged), -1, dtype=np.int8)

    # Fixed: create a writable timestamp array before replacing invalid values.
    if time_col and time_col in merged.columns:
        parsed = pd.to_datetime(
            merged[time_col], errors="coerce", dayfirst=True
        )
        timestamps = parsed.astype("int64").to_numpy(copy=True)
        invalid = parsed.isna().to_numpy()
        timestamps[invalid] = np.arange(
            len(merged), dtype=np.int64
        )[invalid]
    else:
        timestamps = np.arange(len(merged), dtype=np.int64)

    out = out_root / "swat"
    save_npz(
        out / "train_normal.npz",
        X=X_train,
        feature_names=np.asarray(feature_cols),
    )
    save_npz(
        out / "test.npz",
        X=X_test,
        y=y_test,
        timestamps=timestamps,
        feature_names=np.asarray(feature_cols),
    )
    save_npz(
        out / "scaler.npz",
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        feature_names=np.asarray(feature_cols),
    )
    print(
        f"[SWaT] Train shape={X_train.shape}; test shape={X_test.shape}; "
        f"anomaly rows={(y_test == 1).sum()}; output={out}"
    )


def locate_smd_root(root: Path) -> Path:
    """Find directory directly containing train/, test/, and test_label/."""
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for folder in candidates:
        if (
            (folder / "train").is_dir()
            and (folder / "test").is_dir()
            and (folder / "test_label").is_dir()
        ):
            return folder
    raise FileNotFoundError(
        f"Could not locate train/, test/, and test_label/ under {root}. "
        "Check the extracted SMD folder structure."
    )


def machine_files(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.glob("machine-*.txt")}


def load_matrix(path: Path, dtype=np.float32) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=dtype, ndmin=2)


def preprocess_smd(root: Path, out_root: Path) -> None:
    smd_root = locate_smd_root(root)
    train_files = machine_files(smd_root / "train")
    test_files = machine_files(smd_root / "test")
    label_files = machine_files(smd_root / "test_label")
    machine_ids = sorted(train_files.keys())

    if not machine_ids:
        raise FileNotFoundError(
            f"No train/machine-*.txt files found in {smd_root / 'train'}"
        )
    print(f"[SMD] Dataset root: {smd_root}; machines found: {len(machine_ids)}")

    out = out_root / "smd"
    summary = []
    for machine in machine_ids:
        if machine not in test_files or machine not in label_files:
            print(f"[SMD] WARNING: skipping {machine}; test/label file missing.")
            continue

        train = load_matrix(train_files[machine])
        test = load_matrix(test_files[machine])
        labels = load_matrix(label_files[machine], dtype=np.int8).reshape(-1)

        if train.shape[1] != test.shape[1]:
            raise ValueError(f"{machine}: train/test feature dimensions differ.")
        if len(test) != len(labels):
            raise ValueError(
                f"{machine}: test rows ({len(test)}) != label rows ({len(labels)})."
            )

        # Fit scaler per machine on its training split only.
        scaler = StandardScaler()
        train_clean = clean_numeric(pd.DataFrame(train)).to_numpy()
        test_clean = clean_numeric(pd.DataFrame(test)).to_numpy()
        X_train = scaler.fit_transform(train_clean).astype(np.float32)
        X_test = scaler.transform(test_clean).astype(np.float32)

        machine_out = out / machine
        save_npz(machine_out / "train_normal.npz", X=X_train)
        save_npz(machine_out / "test.npz", X=X_test, y=labels)
        save_npz(
            machine_out / "scaler.npz",
            mean_=scaler.mean_,
            scale_=scaler.scale_,
        )
        summary.append(
            {
                "machine": machine,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "n_features": X_train.shape[1],
                "anomaly_rows": int(labels.sum()),
            }
        )
        print(
            f"[SMD] {machine}: train={X_train.shape}, test={X_test.shape}, "
            f"anomalies={int(labels.sum())}"
        )

    out.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "summary.csv", index=False)
    print(f"[SMD] Saved summary to {out / 'summary.csv'}")


class SlidingWindowDataset:
    """Lazy sliding windows: ds[i] returns an array shaped [window, features]."""

    def __init__(
        self, X: np.ndarray, window_size: int = 128, stride: int = 1
    ):
        if X.ndim != 2:
            raise ValueError("X must have shape [time, features].")
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be >= 1.")
        if len(X) < window_size:
            raise ValueError(f"Need at least {window_size} rows, got {len(X)}.")
        self.X = X
        self.window_size = int(window_size)
        self.stride = int(stride)

    def __len__(self):
        return 1 + (len(self.X) - self.window_size) // self.stride

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        start = idx * self.stride
        return self.X[start : start + self.window_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swat-dir", type=Path, default=None)
    parser.add_argument("--smd-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("processed"))
    args = parser.parse_args()

    if args.swat_dir is None and args.smd_dir is None:
        parser.error("Provide --swat-dir and/or --smd-dir.")

    if args.swat_dir:
        if not args.swat_dir.is_dir():
            raise NotADirectoryError(f"SWAT folder not found: {args.swat_dir}")
        preprocess_swat(args.swat_dir, args.out)

    if args.smd_dir:
        if not args.smd_dir.is_dir():
            raise NotADirectoryError(f"SMD folder not found: {args.smd_dir}")
        preprocess_smd(args.smd_dir, args.out)


if __name__ == "__main__":
    main()
