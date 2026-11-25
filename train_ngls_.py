"""
NGLS-inspired latent diffusion anomaly detector.

This is a lightweight adaptation, not an exact reproduction of NGLS-Diff.
It preserves the project's window_dataset.py interface:
  NormalWindowDataset(..., split="train"/"val")
  TestWindowDataset(..., label_mode="last")

Training uses only normal windows. The encoder + normal-pattern gatherer are
trained with reconstruction and regularization losses, then latent diffusion
is trained on the gathered normal representations. Anomaly score combines
latent denoising error and reconstruction error, calibrated on normal validation.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from window_dataset import NormalWindowDataset, TestWindowDataset


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -np.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = t.float()[:, None] * freq[None, :]
        emb = torch.cat([angles.sin(), angles.cos()], dim=1)
        if emb.shape[1] < self.dim:
            emb = nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
        return self.net(emb)


class NormalGatheringAE(nn.Module):
    """
    Approximate NGLS-inspired representation:
    - per-timestep encoder, preserving temporal length;
    - patch-wise position weights over a small bank of normal patterns;
    - pattern-specific shape mean/scale;
    - shared/global feature projection;
    - fusion and decoder back to input space.
    """
    def __init__(self, features, hidden=64, latent=16, patterns=3, patch=10):
        super().__init__()
        if hidden % 8:
            raise ValueError("hidden must be divisible by 8 for GroupNorm")
        self.patch = patch
        self.patterns = patterns
        self.latent = latent
        self.encoder = nn.Sequential(
            nn.Linear(features, hidden), nn.SiLU(),
            nn.Linear(hidden, latent), nn.SiLU(),
        )
        self.patch_encoder = nn.Sequential(
            nn.Conv1d(latent, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.position_head = nn.Conv1d(hidden, patterns, kernel_size=1)
        self.shape_mu = nn.Linear(hidden, patterns * latent)
        self.shape_logvar = nn.Linear(hidden, patterns * latent)
        self.common = nn.Sequential(
            nn.Linear(latent, latent), nn.SiLU(), nn.Linear(latent, latent)
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(patterns * latent + latent, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, latent, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, features)
        )

    def forward(self, x, sample=True):
        # x: [B,T,F], z: [B,T,D]
        z = self.encoder(x)
        b, t, d = z.shape
        pad = (-t) % self.patch
        if pad:
            z_pad = torch.cat([z, z[:, -1:, :].expand(-1, pad, -1)], dim=1)
        else:
            z_pad = z
        tp = z_pad.shape[1]
        npatches = tp // self.patch

        # Summarize each non-overlapping patch; position weights select patterns.
        patches = z_pad.reshape(b, npatches, self.patch, d).mean(dim=2)
        h = self.patch_encoder(patches.transpose(1, 2)).transpose(1, 2)
        pos = torch.softmax(self.position_head(h.transpose(1, 2)), dim=1)
        # [B,P,Npatch]
        mu = self.shape_mu(h).reshape(b, npatches, self.patterns, d)
        logvar = self.shape_logvar(h).reshape(b, npatches, self.patterns, d).clamp(-6, 4)
        if sample:
            shape = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            shape = mu

        # Pattern-weighted shape representation, plus a shared global component.
        weights = pos.transpose(1, 2).unsqueeze(-1)  # [B,Npatch,P,1]
        gathered = (weights * shape).reshape(b, npatches, self.patterns * d)
        common = self.common(patches.mean(dim=1)).unsqueeze(1).expand(-1, npatches, -1)
        patch_latent = self.fuse(torch.cat([gathered, common], dim=-1).transpose(1, 2))
        patch_latent = patch_latent.transpose(1, 2)
        # Restore patch representation to each timestep.
        z_gathered = patch_latent.repeat_interleave(self.patch, dim=1)[:, :t, :]
        recon = self.decoder(z_gathered)
        # A simple regularizer encourages bounded shape distributions.
        kl_shape = 0.5 * (mu.square() + logvar.exp() - logvar - 1).mean()
        pos_entropy = -(pos.clamp_min(1e-8) * pos.clamp_min(1e-8).log()).sum(1).mean()
        return recon, z_gathered, kl_shape, pos_entropy


class LatentDenoiser(nn.Module):
    def __init__(self, latent, hidden=64):
        super().__init__()
        self.time = TimeEmbed(hidden)
        self.inp = nn.Conv1d(latent, hidden, 3, padding=1)
        self.time_proj = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(8, hidden), nn.SiLU(),
            ) for _ in range(3)
        ])
        self.out = nn.Conv1d(hidden, latent, 3, padding=1)

    def forward(self, z, t):
        h = self.inp(z.transpose(1, 2)) + self.time_proj(self.time(t)).unsqueeze(-1)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h).transpose(1, 2)


class Diffusion:
    def __init__(self, steps, device):
        self.steps = steps
        beta = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alpha_bar = torch.cumprod(1.0 - beta, dim=0)

    def corrupt(self, z, t):
        eps = torch.randn_like(z)
        a = self.alpha_bar[t, None, None]
        return a.sqrt() * z + (1 - a).sqrt() * eps, eps

    @torch.no_grad()
    def score(self, denoiser, z, draws=4):
        errors = []
        for _ in range(draws):
            t = torch.randint(0, self.steps, (len(z),), device=z.device)
            zt, eps = self.corrupt(z, t)
            errors.append((denoiser(zt, t) - eps).square().mean((1, 2)))
        return torch.stack(errors).mean(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(train_path, test_path, out_path, args, device, label):
    print(f"\n{'=' * 12} {label} {'=' * 12}", flush=True)
    set_seed(args.seed)

    train_ds = NormalWindowDataset(train_path, args.window, args.stride, "train")
    val_ds = NormalWindowDataset(train_path, args.window, args.stride, "val")
    test_ds = TestWindowDataset(test_path, args.window, args.stride, "last")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    features = train_ds[0].shape[-1]
    ae = NormalGatheringAE(
        features, args.hidden, args.latent, args.patterns, args.patch
    ).to(device)
    denoiser = LatentDenoiser(args.latent, args.hidden).to(device)
    diffusion = Diffusion(args.steps, device)

    # Stage 1: learn a normal-pattern latent representation from normal windows.
    ae_opt = torch.optim.AdamW(ae.parameters(), lr=args.lr)
    for epoch in range(args.ae_epochs):
        ae.train()
        losses = []
        for x in train_loader:
            x = x.to(device)
            recon, _, kl, entropy = ae(x, sample=True)
            recon_loss = (recon - x).square().mean()
            loss = recon_loss + args.kl_weight * kl + args.pos_weight * entropy
            ae_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ae_opt.step()
            losses.append(float(loss.detach()))
        print(f"AE epoch {epoch+1}/{args.ae_epochs} loss={np.mean(losses):.6f}", flush=True)

    # Freeze representation while fitting diffusion to normal gathered latents.
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    diff_opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr)
    for epoch in range(args.diff_epochs):
        denoiser.train()
        losses = []
        for x in train_loader:
            x = x.to(device)
            with torch.no_grad():
                _, z, _, _ = ae(x, sample=False)
            t = torch.randint(0, args.steps, (len(x),), device=device)
            zt, eps = diffusion.corrupt(z, t)
            loss = (denoiser(zt, t) - eps).square().mean()
            diff_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            diff_opt.step()
            losses.append(float(loss.detach()))
        print(f"diffusion epoch {epoch+1}/{args.diff_epochs} loss={np.mean(losses):.6f}", flush=True)

    @torch.no_grad()
    def batch_scores(x):
        recon, z, _, _ = ae(x, sample=False)
        latent_error = diffusion.score(denoiser, z, args.draws)
        recon_error = (recon - x).square().mean((1, 2))
        if args.score_mode == "latent":
            return latent_error
        if args.score_mode == "reconstruction":
            return recon_error
        return latent_error + args.recon_weight * recon_error

    denoiser.eval()
    val_scores = np.concatenate([batch_scores(x.to(device)).cpu().numpy() for x in val_loader])
    threshold = float(np.percentile(val_scores, args.threshold_percentile))

    scores, labels = [], []
    elapsed, count = 0.0, 0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            score = batch_scores(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            count += len(x)
            scores.append(score.cpu().numpy())
            labels.append(y.numpy())

    scores = np.concatenate(scores)
    labels = np.concatenate(labels).astype(int)
    pred = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    pr_auc = average_precision_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")
    latency_ms = 1000 * elapsed / max(count, 1)

    print(f"threshold={threshold:.6f} | precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}")
    print(f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}")
    print(f"scoring latency={latency_ms:.3f} ms/window (batched)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ae": ae.state_dict(),
        "denoiser": denoiser.state_dict(),
        "threshold": threshold,
        "window": args.window,
        "features": features,
        "args": vars(args),
        "dataset": label,
    }, out_path)
    print("Saved checkpoint:", out_path)
    return {
        "dataset": label, "train_file": str(train_path), "test_file": str(test_path),
        "threshold": threshold, "threshold_percentile": args.threshold_percentile,
        "test_windows": len(labels), "anomaly_windows": int(labels.sum()),
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "latency_ms_per_window": latency_ms, "checkpoint": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="NGLS-inspired latent diffusion anomaly detector.")
    parser.add_argument("--train", default="processed/swat/train_normal.npz")
    parser.add_argument("--test", default="processed/swat/test.npz")
    parser.add_argument("--all-smd", action="store_true")
    parser.add_argument("--smd-dir", default="processed/smd")
    parser.add_argument("--results-csv", default="runs/smd/ngls_inspired_results.csv")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--patch", type=int, default=10)
    parser.add_argument("--patterns", type=int, default=3)
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--diff-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=1e-3)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--score-mode", choices=["latent", "reconstruction", "combined"], default="combined")
    parser.add_argument("--threshold-percentile", type=float, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/ngls_inspired_swat.pt")
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
        for i, folder in enumerate(folders):
            args.seed = 42 + i
            checkpoint = Path(args.out).parent / "smd" / f"{folder.name}_ngls_inspired.pt"
            rows.append(train_one(folder / "train_normal.npz", folder / "test.npz",
                                  checkpoint, args, device, folder.name))
            result_path = Path(args.results_csv)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    else:
        train_one(args.train, args.test, args.out, args, device, "single-run")


if __name__ == "__main__":
    main()
