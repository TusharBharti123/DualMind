import argparse, os, time, random, csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from window_dataset import NormalWindowDataset, TestWindowDataset


class TimeEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(), nn.Linear(d * 2, d))

    def forward(self, t):
        h = self.d // 2
        f = torch.exp(-np.log(10000) * torch.arange(
            h, device=t.device, dtype=torch.float32) / max(h - 1, 1))
        a = t.float()[:, None] * f[None, :]
        e = torch.cat([a.sin(), a.cos()], 1)
        if e.shape[1] < self.d:
            e = nn.functional.pad(e, (0, self.d - e.shape[1]))
        return self.net(e)


class Denoiser(nn.Module):
    def __init__(self, features, hidden=64):
        super().__init__()
        self.te = TimeEmbed(hidden)
        self.inp = nn.Conv1d(features, hidden, 3, padding=1)
        self.tp = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1),
                          nn.GroupNorm(8, hidden), nn.SiLU())
            for _ in range(3)
        ])
        self.out = nn.Conv1d(hidden, features, 3, padding=1)

    def forward(self, x, t):
        h = self.inp(x.transpose(1, 2)) + self.tp(self.te(t)).unsqueeze(-1)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h).transpose(1, 2)


class Diffusion:
    def __init__(self, n, device):
        self.n = n
        beta = torch.linspace(1e-4, .02, n, device=device)
        self.ab = torch.cumprod(1 - beta, 0)

    def corrupt(self, x, t):
        eps = torch.randn_like(x)
        a = self.ab[t, None, None]
        return a.sqrt() * x + (1 - a).sqrt() * eps, eps

    @torch.no_grad()
    def score(self, model, x, draws):
        vals = []
        for _ in range(draws):
            t = torch.randint(0, self.n, (len(x),), device=x.device)
            xt, eps = self.corrupt(x, t)
            vals.append((model(xt, t) - eps).square().mean((1, 2)))
        return torch.stack(vals).mean(0)


def train_one(train_path, test_path, out_path, args, device, label):
    print(f"\\n{'=' * 12} {label} {'=' * 12}", flush=True)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tr = NormalWindowDataset(str(train_path), args.window, args.stride, "train")
    va = NormalWindowDataset(str(train_path), args.window, args.stride, "val")
    te = TestWindowDataset(str(test_path), args.window, args.stride, "last")
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=args.batch_size)
    test_loader = DataLoader(te, batch_size=args.batch_size)

    model = Denoiser(tr[0].shape[-1]).to(device)
    diff = Diffusion(args.steps, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        losses = []
        for x in train_loader:
            x = x.to(device)
            t = torch.randint(0, args.steps, (len(x),), device=device)
            xt, eps = diff.corrupt(x, t)
            loss = (model(xt, t) - eps).square().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep + 1}/{args.epochs} train_mse={np.mean(losses):.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        val_scores = np.concatenate([
            diff.score(model, x.to(device), args.draws).cpu().numpy()
            for x in val_loader
        ])
    threshold = float(np.percentile(val_scores, args.threshold_percentile))
    print("normal validation threshold:", threshold)

    scores, labels = [], []
    elapsed, count = 0.0, 0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            score = diff.score(model, x, args.draws)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
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

    print(f"test windows={len(labels)} anomalies={labels.sum()} "
          f"precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}")
    if np.isfinite(roc_auc):
        print(f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}")
    print(f"scoring latency={latency_ms:.3f} ms/window (batched)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "threshold": threshold,
        "window": args.window,
        "features": tr[0].shape[-1],
        "args": vars(args),
        "dataset": label,
    }, out_path)
    print("Saved checkpoint:", out_path)

    return {
        "dataset": label,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "test_windows": len(labels),
        "anomaly_windows": int(labels.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "latency_ms_per_window": latency_ms,
        "checkpoint": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate diffusion anomaly detector on one dataset or all SMD machines."
    )
    parser.add_argument("--train", default="processed/swat/train_normal.npz")
    parser.add_argument("--test", default="processed/swat/test.npz")
    parser.add_argument("--all-smd", action="store_true",
                        help="Run every machine folder under --smd-dir sequentially.")
    parser.add_argument("--smd-dir", default="processed/smd",
                        help="Directory containing machine-* folders.")
    parser.add_argument("--results-csv", default="runs/smd/results.csv",
                        help="Summary CSV used with --all-smd.")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--threshold-percentile", type=float, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/diffusion_swat.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    if args.all_smd:
        smd_root = Path(args.smd_dir)
        machine_dirs = sorted(
            p for p in smd_root.glob("machine-*")
            if p.is_dir() and (p / "train_normal.npz").exists() and (p / "test.npz").exists()
        )
        if not machine_dirs:
            raise SystemExit(f"No machine-* folders with train_normal.npz and test.npz found in {smd_root}")

        results = []
        for index, machine_dir in enumerate(machine_dirs):
            # Distinct but reproducible seed per machine.
            args.seed = int(parser.get_default("seed")) + index
            checkpoint = Path(args.out).parent / "smd" / f"{machine_dir.name}.pt"
            try:
                row = train_one(
                    machine_dir / "train_normal.npz",
                    machine_dir / "test.npz",
                    checkpoint,
                    args,
                    device,
                    machine_dir.name,
                )
                results.append(row)
                
                results_path = Path(args.results_csv)
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with results_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)
            except Exception as exc:
                print(f"FAILED {machine_dir.name}: {exc}", flush=True)
                raise
        print(f"\\nCompleted {len(results)} SMD machines. Summary: {args.results_csv}")
    else:
        train_one(args.train, args.test, args.out, args, device, "single-run")


if __name__ == "__main__":
    main()
