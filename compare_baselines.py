"""
Head-to-Head Comparison: Latent-Space Diffusion Baseline (train_ngls_.py) vs.
One-Class Label-Space Diffusion (train_label_diffusion.py) on identical test splits,
including single-window latency relative to sensor arrival interval.
Team DualMind — Tushar Bharti & Aditya Gupta
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from window_dataset import NormalWindowDataset, TestWindowDataset
from train_ngls_ import NormalGatheringAE, LatentDenoiser, Diffusion as LatentDiff
from label_space_model import OneClassLabelDiffusion


def measure_single_window_ms(fn, sample_x: torch.Tensor, device: torch.device, iters: int = 100) -> float:
    for _ in range(15):
        _ = fn(sample_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(sample_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / iters


def _get_cfg(ckpt: dict, args_dict: dict, key: str, default):
    if key in args_dict and args_dict[key] is not None:
        return args_dict[key]
    if key in ckpt and ckpt[key] is not None:
        return ckpt[key]
    return default


@torch.no_grad()
def eval_latent_baseline(ckpt_path: Path, train_path: Path, test_path: Path, stride: int, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args") or {}

    features = ckpt.get("features")
    if features is None and "ae" in ckpt:
        first_tensor = next(iter(ckpt["ae"].values()))
        features = int(first_tensor.shape[-1])

    hidden = _get_cfg(ckpt, a, "hidden", 64)
    latent = _get_cfg(ckpt, a, "latent", 16)
    patterns = _get_cfg(ckpt, a, "patterns", 3)
    patch = _get_cfg(ckpt, a, "patch", 10)
    steps = _get_cfg(ckpt, a, "steps", 100)
    draws = _get_cfg(ckpt, a, "draws", 4)
    score_mode = _get_cfg(ckpt, a, "score_mode", "combined")
    recon_weight = _get_cfg(ckpt, a, "recon_weight", 1.0)
    thresh_pct = _get_cfg(ckpt, a, "threshold_percentile", 99.0)
    window = ckpt.get("window", 100)

    ae = NormalGatheringAE(features, hidden, latent, patterns, patch).to(device)
    denoiser = LatentDenoiser(latent, hidden).to(device)
    ae.load_state_dict(ckpt["ae"])
    denoiser.load_state_dict(ckpt["denoiser"])
    ae.eval()
    denoiser.eval()
    diff = LatentDiff(steps, device)

    def score_fn(x: torch.Tensor) -> torch.Tensor:
        recon, z, _, _ = ae(x, sample=False)
        lat_err = diff.score(denoiser, z, draws)
        rec_err = (recon - x).square().mean((1, 2))
        if score_mode == "latent":
            return lat_err
        if score_mode == "reconstruction":
            return rec_err
        return lat_err + recon_weight * rec_err

    val_ds = NormalWindowDataset(str(train_path), window, stride, split="val", val_ratio=0.1)
    test_ds = TestWindowDataset(str(test_path), window, stride, label_mode="last")
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    val_scores = np.concatenate([score_fn(x.to(device)).cpu().numpy() for x in val_loader])
    thresh = float(np.percentile(val_scores, thresh_pct))

    scores, labels = [], []
    for x, y, _ in test_loader:
        scores.append(score_fn(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(int)
    pred = (scores > thresh).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    roc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    pr = average_precision_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    lat_ms = measure_single_window_ms(score_fn, test_ds[0][0].unsqueeze(0).to(device), device)

    return {
        "method": "Latent-Space Diffusion (Baseline)",
        "threshold": thresh,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "inference_ms": lat_ms,
    }


@torch.no_grad()
def eval_label_space(ckpt_path: Path, train_path: Path, test_path: Path, stride: int, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args") or {}

    model = OneClassLabelDiffusion(
        in_features=_get_cfg(ckpt, a, "features", None),
        num_regimes=_get_cfg(ckpt, a, "num_regimes", 12),
        hidden=_get_cfg(ckpt, a, "hidden", 128),
        steps=_get_cfg(ckpt, a, "steps", 100),
        ddim_steps=_get_cfg(ckpt, a, "ddim_steps", 4),
        num_seeds=_get_cfg(ckpt, a, "num_seeds", 8),
        var_weight=_get_cfg(ckpt, a, "var_weight", 2.0),
        temperature=_get_cfg(ckpt, a, "temperature", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    score_fn = lambda x: model.anomaly_score(x)[0]
    window = ckpt.get("window", 100)
    thresh_pct = _get_cfg(ckpt, a, "threshold_percentile", 99.0)

    val_ds = NormalWindowDataset(str(train_path), window, stride, split="val", val_ratio=0.1)
    test_ds = TestWindowDataset(str(test_path), window, stride, label_mode="last")
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    val_scores = np.concatenate([score_fn(x.to(device)).cpu().numpy() for x in val_loader])
    thresh = float(np.percentile(val_scores, thresh_pct))

    scores, labels = [], []
    for x, y, _ in test_loader:
        scores.append(score_fn(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(int)
    pred = (scores > thresh).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    roc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    pr = average_precision_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    lat_ms = measure_single_window_ms(score_fn, test_ds[0][0].unsqueeze(0).to(device), device)

    return {
        "method": "One-Class Label-Space Diffusion (Ours)",
        "threshold": thresh,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "inference_ms": lat_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["SWaT", "SMD"], default="SWaT")
    ap.add_argument("--train", default="processed/swat/train_normal.npz")
    ap.add_argument("--test", default="processed/swat/test.npz")
    ap.add_argument("--latent-ckpt", default="runs/ngls_inspired_swat.pt")
    ap.add_argument("--label-ckpt", default="runs/label_diff_swat.pt")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--arrival-interval-ms", type=float, default=None,
                    help="Inter-window arrival interval in ms (defaults: SWaT=1000ms at stride=1, SMD=60000ms)")
    ap.add_argument("--out-csv", default="runs/comparison_swat.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SWaT samples at 1 Hz (1000 ms/tick); SMD samples at 1/60 Hz (60000 ms/tick)
    tick_ms = 1000.0 if args.dataset == "SWaT" else 60000.0
    arrival_ms = args.arrival_interval_ms if args.arrival_interval_ms is not None else tick_ms

    rows = [
        eval_latent_baseline(Path(args.latent_ckpt), Path(args.train), Path(args.test), args.stride, device),
        eval_label_space(Path(args.label_ckpt), Path(args.train), Path(args.test), args.stride, device),
    ]

    for r in rows:
        r["dataset"] = args.dataset
        r["arrival_interval_ms"] = arrival_ms
        r["realtime_ratio"] = r["inference_ms"] / arrival_ms
        r["realtime_speedup_factor"] = arrival_ms / max(r["inference_ms"], 1e-6)
        r["is_realtime"] = r["inference_ms"] < arrival_ms

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== {args.dataset} Head-to-Head Comparison (Calibrated on Held-Out Normal Val) ===")
    for r in rows:
        print(
            f"{r['method']:40s} | F1={r['f1']:.4f} | ROC-AUC={r['roc_auc']:.4f} | PR-AUC={r['pr_auc']:.4f} | "
            f"t_inf={r['inference_ms']:.3f} ms vs Delta_t={r['arrival_interval_ms']:.0f} ms "
            f"(Ratio={r['realtime_ratio']:.5f}, {r['realtime_speedup_factor']:.0f}x faster)"
        )
    print(f"\nSaved comparison CSV to: {out_path.resolve()}")


if __name__ == "__main__":
    main()