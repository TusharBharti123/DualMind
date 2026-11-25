"""
Train & Evaluate One-Class Label-Space Diffusion on SWaT and SMD.


Usage:
  # SWaT
  python train_label_diffusion.py --train processed/swat/train_normal.npz --test processed/swat/test.npz --out runs/label_diff_swat.pt

  # All 28 SMD machines
  python train_label_diffusion.py --all-smd --smd-dir processed/smd --results-csv runs/smd/label_diff_results.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from window_dataset import NormalWindowDataset, TestWindowDataset
from label_space_model import OneClassLabelDiffusion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def affiliation_or_point_metrics(labels: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    pr_auc = average_precision_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def train_one(
    train_path: str | Path,
    test_path: str | Path,
    out_path: str | Path,
    args: argparse.Namespace,
    device: torch.device,
    label: str,
) -> dict:
    print(f"\n{'=' * 14} {label} (One-Class Label-Space Diffusion) {'=' * 14}", flush=True)
    set_seed(args.seed)

    # Chronological 90/10 split on normal-only training data (val is strictly held-out normal)
    train_ds = NormalWindowDataset(str(train_path), args.window, args.stride, split="train", val_ratio=0.1)
    val_ds = NormalWindowDataset(str(train_path), args.window, args.stride, split="val", val_ratio=0.1)
    test_ds = TestWindowDataset(str(test_path), args.window, args.stride, label_mode="last")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    features = train_ds[0].shape[-1]
    model = OneClassLabelDiffusion(
        in_features=features,
        num_regimes=args.num_regimes,
        hidden=args.hidden,
        steps=args.steps,
        ddim_steps=args.ddim_steps,
        num_seeds=args.num_seeds,
        var_weight=args.var_weight,
        temperature=args.temperature,
    ).to(device)

    # ------------------------------------------------------------------
    # Stage 1: Self-supervised TCN pretraining on normal windows only
    # ------------------------------------------------------------------
    enc_opt = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr)
    for ep in range(args.enc_epochs):
        model.encoder.train()
        losses = []
        for x in train_loader:
            x = x.to(device)
            # Predict next-step sensor readings from causal history [0..L-2] -> [1..L-1]
            h_seq = model.encoder.encode_sequence(x[:, :-1, :])
            pred_next = model.encoder.forecast_head(h_seq)
            loss = F.mse_loss(pred_next, x[:, 1:, :])
            enc_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            enc_opt.step()
            losses.append(float(loss.detach()))
        print(f"Stage 1 (TCN Encoder) epoch {ep + 1}/{args.enc_epochs} forecast_mse={np.mean(losses):.5f}", flush=True)

    # ------------------------------------------------------------------
    # Stage 2: Discover K Normal Operating Regime Prototypes on train_ds
    # ------------------------------------------------------------------
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        all_c = torch.cat([model.encoder(x.to(device)) for x in train_loader], dim=0).cpu().numpy()
    model.bank.fit(all_c, seed=args.seed)
    print(f"Stage 2 (Regime Prototypes) fitted K={args.num_regimes} normal prototypes on {len(all_c):,} windows.", flush=True)

    # ------------------------------------------------------------------
    # Stage 3: Train Conditional Label-Space Denoiser on y_0 in Delta^{K-1}
    # ------------------------------------------------------------------
    diff_opt = torch.optim.AdamW(model.denoiser.parameters(), lr=args.lr)
    for ep in range(args.diff_epochs):
        model.denoiser.train()
        losses = []
        for x in train_loader:
            x = x.to(device)
            loss = model.diffusion_loss(x)
            diff_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
            diff_opt.step()
            losses.append(float(loss.detach()))
        print(f"Stage 3 (Label Denoiser) epoch {ep + 1}/{args.diff_epochs} label_loss={np.mean(losses):.6f}", flush=True)

    # ------------------------------------------------------------------
    # Stage 4: Held-Out Normal Validation Threshold Calibration
    # ------------------------------------------------------------------
    model.eval()
    val_scores = np.concatenate([
        model.anomaly_score(x.to(device))[0].cpu().numpy()
        for x in val_loader
    ])
    threshold = float(np.percentile(val_scores, args.threshold_percentile))
    print(f"Stage 4 (Held-out Normal Val) {args.threshold_percentile}th percentile threshold = {threshold:.6f}", flush=True)

    # ------------------------------------------------------------------
    # Stage 5: Test Evaluation & Streaming Latency Measurement
    # ------------------------------------------------------------------
    scores, labels = [], []
    elapsed, count = 0.0, 0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s, _ = model.anomaly_score(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            count += len(x)
            scores.append(s.cpu().numpy())
            labels.append(y.numpy())

    # Single-window (batch=1) real-time streaming latency check
    single_x = test_ds[0][0].unsqueeze(0).to(device)
    for _ in range(10):  # warmup
        _ = model.anomaly_score(single_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_single0 = time.perf_counter()
    for _ in range(100):
        _ = model.anomaly_score(single_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_window_ms = 1000.0 * (time.perf_counter() - t_single0) / 100.0

    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(int)
    pred = (scores > threshold).astype(int)
    metrics = affiliation_or_point_metrics(labels, pred, scores)
    batched_ms = 1000.0 * elapsed / max(count, 1)

    print(
        f"Test windows={len(labels)} anomalies={labels.sum()} | "
        f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}",
        flush=True,
    )
    print(f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}", flush=True)
    print(
        f"Latency: single-window={single_window_ms:.3f} ms | batched={batched_ms:.4f} ms/window",
        flush=True,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "threshold": threshold,
        "window": args.window,
        "features": features,
        "num_regimes": args.num_regimes,
        "hidden": args.hidden,
        "steps": args.steps,
        "ddim_steps": args.ddim_steps,
        "num_seeds": args.num_seeds,
        "var_weight": args.var_weight,
        "temperature": args.temperature,
        "args": vars(args),
        "dataset": label,
    }, out_path)
    print("Saved checkpoint:", out_path, flush=True)

    return {
        "dataset": label,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "test_windows": len(labels),
        "anomaly_windows": int(labels.sum()),
        **metrics,
        "single_window_ms": single_window_ms,
        "batched_ms_per_window": batched_ms,
        "checkpoint": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train One-Class Label-Space Diffusion detector.")
    parser.add_argument("--train", default="processed/swat/train_normal.npz")
    parser.add_argument("--test", default="processed/swat/test.npz")
    parser.add_argument("--all-smd", action="store_true")
    parser.add_argument("--smd-dir", default="processed/smd")
    parser.add_argument("--results-csv", default="runs/smd/label_diff_results.csv")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--num-regimes", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--enc-epochs", type=int, default=8)
    parser.add_argument("--diff-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--var-weight", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/label_diff_swat.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    if args.all_smd:
        root = Path(args.smd_dir)
        folders = sorted(
            p for p in root.glob("machine-*")
            if p.is_dir() and (p / "train_normal.npz").exists() and (p / "test.npz").exists()
        )
        if not folders:
            raise SystemExit(f"No valid machine-* folders found in {root}")
        rows = []
        for idx, folder in enumerate(folders):
            args.seed = 42 + idx
            ckpt = Path(args.out).parent / "smd" / f"{folder.name}_label_diff.pt"
            rows.append(train_one(
                folder / "train_normal.npz",
                folder / "test.npz",
                ckpt,
                args,
                device,
                folder.name,
            ))
            res_path = Path(args.results_csv)
            res_path.parent.mkdir(parents=True, exist_ok=True)
            with res_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"\nCompleted {len(rows)} SMD machines -> {args.results_csv}")
    else:
        train_one(args.train, args.test, args.out, args, device, "SWaT")


if __name__ == "__main__":
    main()