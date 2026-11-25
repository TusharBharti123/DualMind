"""
Step-by-Step Trace of a Normal Window vs. Anomalous Window through
One-Class Label-Space Diffusion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from window_dataset import TestWindowDataset
from label_space_model import OneClassLabelDiffusion


def load_model(ckpt_path: Path, device: torch.device) -> tuple[OneClassLabelDiffusion, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = OneClassLabelDiffusion(
        in_features=ckpt["features"],
        num_regimes=ckpt["num_regimes"],
        hidden=ckpt["hidden"],
        steps=ckpt["steps"],
        ddim_steps=ckpt["ddim_steps"],
        num_seeds=ckpt["num_seeds"],
        var_weight=ckpt["var_weight"],
        temperature=ckpt["temperature"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt["threshold"])


def find_example_windows(ds: TestWindowDataset) -> tuple[int, int]:
    normal_idx, anom_idx = None, None
    for i in range(len(ds)):
        _, y, _ = ds[i]
        if int(y) == 0 and normal_idx is None:
            normal_idx = i
        elif int(y) == 1 and anom_idx is None:
            anom_idx = i
        if normal_idx is not None and anom_idx is not None:
            break
    if normal_idx is None or anom_idx is None:
        raise RuntimeError("Could not find both a normal (0) and anomalous (1) window in test set.")
    return normal_idx, anom_idx


def trace_single_window(
    model: OneClassLabelDiffusion,
    x: torch.Tensor,
    threshold: float,
    tag: str,
) -> dict:
    # x: [1, L, D]
    c = model.encoder(x)
    y_ref = model.bank(c)
    y_samples, traj = model.ddim_sample_from_context(c, return_trajectories=True)
    total_score, details = model.anomaly_score(x)

    y_ref_np = y_ref[0].detach().cpu().numpy()
    y_samples_np = y_samples[0].cpu().numpy()  # [M, K]
    y_noise_np = traj[0][0].cpu().numpy()       # [M, K] initial Gaussian noise in label space
    jsd_val = float(details["jsd_mismatch"][0].cpu())
    var_val = float(details["seed_variance"][0].cpu())
    tot_val = float(total_score[0].cpu())
    pred = int(tot_val > threshold)

    print(f"\n{'-' * 68}")
    print(f"[{tag}] Input sensor window X shape: {tuple(x.shape)} (UNTOUCHED — never noised)")
    print(f"  1. Condition vector c = Encoder(X) shape: {tuple(c.shape)}, L2 norm={float(c.norm()):.3f}")
    print(f"  2. Reference normal regime label y_ref in Delta^{model.num_regimes - 1}:")
    print(f"     {np.round(y_ref_np, 3)} (Dominant Regime = R{int(np.argmax(y_ref_np))})")
    print(f"  3. What receives noise -> Label vector y_T ~ N(0, I_{model.num_regimes}) across M={model.num_seeds} seeds:")
    print(f"     Seed 0 initial noise y_T[0]: {np.round(y_noise_np[0], 3)}")
    print(f"  4. Denoised outputs y_hat_0^(m) after {model.ddim_steps} DDIM steps:")
    for m in range(min(4, model.num_seeds)):
        print(f"     Seed {m}: {np.round(y_samples_np[m], 3)} -> Top Regime R{int(np.argmax(y_samples_np[m]))}")
    print(f"  5. Anomaly Score Computation:")
    print(f"     - Regime Consensus Mismatch (JSD):   {jsd_val:.5f}")
    print(f"     - Multi-Seed Denoising Variance:     {var_val:.5f}")
    print(f"     - Total Score S(X) = JSD + {model.var_weight}*Var: {tot_val:.5f}  (Threshold = {threshold:.5f})")
    print(f"     -> Prediction: {'ANOMALY (1)' if pred == 1 else 'NORMAL (0)'}")

    return {
        "x": x[0].cpu().numpy(),
        "y_ref": y_ref_np,
        "y_samples": y_samples_np,
        "jsd": jsd_val,
        "var": var_val,
        "score": tot_val,
        "pred": pred,
    }


def save_trace_figure(norm_res: dict, anom_res: dict, threshold: float, out_fig: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    k = len(norm_res["y_ref"])
    regimes = np.arange(k)

    for row, (title, res, color) in enumerate([
        ("Normal Window (Ground Truth = 0)", norm_res, "#1f77b4"),
        ("Anomalous Window (Ground Truth = 1)", anom_res, "#d62728"),
    ]):
        # Col 0: Raw sensor subset (conditioning input X)
        ax0 = axes[row, 0]
        for ch in range(min(5, res["x"].shape[1])):
            ax0.plot(res["x"][:, ch], alpha=0.8, linewidth=1.2)
        ax0.set_title(f"{title}\n1. Conditioning Input X (Sensors Untouched)")
        ax0.set_xlabel("Timestep in Window")
        ax0.set_ylabel("Z-Normalized Value")

        # Col 1: Reference regime distribution vs denoised multi-seed outputs
        ax1 = axes[row, 1]
        ax1.bar(regimes - 0.18, res["y_ref"], width=0.35, label="Ref Regime y_ref", color="#2ca02c", alpha=0.85)
        mean_hat = res["y_samples"].mean(axis=0)
        std_hat = res["y_samples"].std(axis=0)
        ax1.bar(regimes + 0.18, mean_hat, width=0.35, yerr=std_hat, capsize=3,
                label="Denoised Mean ± Seed Std", color=color, alpha=0.85)
        ax1.set_title("2. Label Space Output (K Normal Regimes)")
        ax1.set_xlabel("Normal Operating Regime ID")
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=8)

        # Col 2: Individual seed trajectories across regimes (heatmap)
        ax2 = axes[row, 2]
        im = ax2.imshow(res["y_samples"], aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax2.set_title(
            f"3. Multi-Seed Denoised Labels [M x K]\n"
            f"Score S(X)={res['score']:.4f} vs Thresh={threshold:.4f}"
        )
        ax2.set_xlabel("Regime ID")
        ax2.set_ylabel("Noise Seed m")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=200)
    print(f"\nSaved visual trace diagram to: {out_fig.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/label_diff_swat.pt")
    ap.add_argument("--test", default="processed/swat/test.npz")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out-fig", default="runs/novelty_trace.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model, threshold = load_model(Path(args.ckpt), device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ds = TestWindowDataset(args.test, window_size=ckpt["window"], stride=args.stride, label_mode="last")
    norm_idx, anom_idx = find_example_windows(ds)

    x_norm = ds[norm_idx][0].unsqueeze(0).to(device)
    x_anom = ds[anom_idx][0].unsqueeze(0).to(device)

    norm_res = trace_single_window(model, x_norm, threshold, f"NORMAL WINDOW (idx={norm_idx})")
    anom_res = trace_single_window(model, x_anom, threshold, f"ANOMALOUS WINDOW (idx={anom_idx})")
    save_trace_figure(norm_res, anom_res, threshold, Path(args.out_fig))


if __name__ == "__main__":
    main()