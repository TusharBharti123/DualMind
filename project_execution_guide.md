# One-Class Label-Space Diffusion for Real-Time Multivariate Time-Series Anomaly Detection

**IE 643 (Deep Learning – Theory and Practice) Course Project — IIT Bombay**

**Team DualMind:** Tushar Bharti (`25M2117`), Aditya Gupta (`25N0461`)

## Overview

This repository provides the end-to-end execution pipeline for our project, following the actual implementation order:

1. **Data Preprocessing (`preprocess.py`):** Chronological normal-only Z-normalization and sliding windowing ($L = 100$, $\text{stride} = 10$) on **SWaT** ($51$ sensors/actuators, $1\text{ Hz}$ / $\Delta t = 1,000\text{ ms}$) and **SMD** ($28$ machines, $38$ metrics each, $1/60\text{ Hz}$ / $\Delta t = 60,000\text{ ms}$).
2. **Latent-Space Diffusion (NGLS) (`train_ngls_.py`):** Implemented first—a Normal-Gathering Autoencoder ($d_z = 16$) paired with a 1D-Conv latent denoiser and combined latent/reconstruction scoring.
3. **One-Class Label-Space Diffusion (`train_label_diffusion.py`):** Implemented next—raw sensor windows $\mathbf{X} \in \mathbb{R}^{L \times D}$ remain un-noised and are encoded once by a Causal Dilated TCN into $\mathbf{c} = E_\phi(\mathbf{X}) \in \mathbb{R}^{128}$, while forward and reverse Gaussian diffusion run strictly on a $K = 12$ normal operating-regime simplex $\mathbf{y}_0 \in \Delta^{K-1}$ using a 4-step deterministic DDIM FiLM denoiser across $M = 8$ noise seeds.
4. **Comparative Evaluation & Novelty Traces (`compare_baselines.py`, `trace_windows.py`):** Head-to-head accuracy and streaming latency benchmarking across SWaT and all 28 SMD machines, plus window-level regime/seed consensus traces.

## 1. Repository Structure

```text
swat_smd_anomaly_detection/
├── data/                         # Raw SWaT and SMD files
├── processed/
│   ├── swat/                     # train_normal.npz, test.npz
│   └── smd/machine-1-1/ ...      # Per-machine train_normal.npz, test.npz (28 machines)
├── runs/
│   ├── swat/                     # SWaT checkpoints, novelty traces, comparison_swat.csv
│   └── smd/                      # Per-machine checkpoints, comparison CSVs, trace PNGs
├── preprocess.py                 # Chronological windowing & normal-only Z-normalization
├── train_ngls_.py                # Trains & evaluates Latent-Space Diffusion (NGLS)
├── train_label_diffusion.py      # Trains & evaluates One-Class Label-Space Diffusion
├── compare_baselines.py          # Head-to-head accuracy & real-time latency benchmark
├── trace_windows.py              # Generates window-level regime & seed consensus traces
└── README.md
```

## 2. Environment Setup

Create and activate the Conda environment with PyTorch and scientific dependencies:

```powershell
conda create -n SMD python=3.10 -y
conda activate SMD

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn scipy matplotlib tqdm openpyxl
```

## 3. Step 1: Data Preprocessing

Both datasets are windowed with length $L = 100$ and $\text{stride} = 10$, applying Z-score normalization fit strictly on normal training data so there is no test leakage.

### 3.1 SWaT Dataset

Place raw SWaT files under `data/swat/` and run:

```powershell
python preprocess.py `
  --dataset swat `
  --raw-dir data/swat `
  --out-dir processed/swat `
  --window 100 `
  --stride 10
```

### 3.2 SMD Dataset (All 28 Machines)

Place SMD files under `data/smd/` and preprocess all 28 machines:

```powershell
python preprocess.py `
  --dataset smd `
  --raw-dir data/smd `
  --out-dir processed/smd `
  --window 100 `
  --stride 10
```

This creates `train_normal.npz` and `test.npz` in `processed/swat/` and each `processed/smd/machine-*/` directory.

## 4. Step 2: Latent-Space Diffusion (NGLS) — Initial Implementation

We first implemented and ran **Latent-Space Diffusion (NGLS)** using a Normal-Gathering Autoencoder ($d_z = 16$) and a 1D-Conv latent denoiser on a chronological `90/10` normal train/validation split with a 99th-percentile validation threshold $\tau_{\text{val}}$.

### 4.1 Train on SWaT

```powershell
python train_ngls_.py `
  --train processed/swat/train_normal.npz `
  --test processed/swat/test.npz `
  --out runs/swat/swat_ngls_inspired.pt
```

### 4.2 Train Across All 28 SMD Machines

```powershell
python train_ngls_.py `
  --all-smd `
  --smd-dir processed/smd `
  --out runs/ngls.pt `
  --results-csv runs/smd/ngls_results.csv
```

*Per-machine latent checkpoints are written to `runs/smd/<machine>_ngls_inspired.pt` and summary metrics to `runs/smd/ngls_results.csv`.*

## 5. Step 3: One-Class Label-Space Diffusion — Subsequent Implementation

After establishing the latent-space NGLS pipeline, we implemented **One-Class Label-Space Diffusion** on the same chronological `90/10` normal splits:

1. Pretrains a 3-block **Causal Dilated TCN** encoder $E_\phi$ via next-step forecasting MSE on normal windows to produce $\mathbf{c} = E_\phi(\mathbf{X}) \in \mathbb{R}^{128}$.
2. Discovers $K = 12$ normal operating-regime prototypes via Spherical K-Means ($\tau = 0.15$), mapping $\mathbf{c}$ to $\mathbf{y}_0 = \mathrm{softmax}(\mathbf{c}^\top \mathbf{M} / \tau) \in \Delta^{K-1}$.
3. Trains a 3-block **FiLM Label Denoiser** $f_\theta(\mathbf{y}_t, t, \mathbf{c})$ with $\mathcal{L}_{\text{diff}} = \|\hat{\mathbf{y}}_0 - \mathbf{y}_0\|_2^2 + 0.1 D_{\mathrm{KL}}(\mathbf{y}_0 \parallel \hat{\mathbf{y}}_0)$ and calibrates the 99th-percentile normal validation threshold $\tau_{\text{val}}$.

### 5.1 Train on SWaT

```powershell
python train_label_diffusion.py `
  --train processed/swat/train_normal.npz `
  --test processed/swat/test.npz `
  --num-regimes 12 `
  --ddim-steps 4 `
  --num-seeds 8 `
  --out runs/swat/swat_label_diff.pt
```

### 5.2 Train Across All 28 SMD Machines

```powershell
python train_label_diffusion.py `
  --all-smd `
  --smd-dir processed/smd `
  --num-regimes 12 `
  --ddim-steps 4 `
  --num-seeds 8 `
  --out runs/smd/label_diff.pt `
  --results-csv runs/smd/label_diff_results.csv
```

*Per-machine label-diffusion checkpoints are written to `runs/smd/<machine>_label_diff.pt` and summary metrics to `runs/smd/label_diff_results.csv`.*

## 6. Step 4: Head-to-Head Evaluation & Real-Time Latency Benchmark

Run `compare_baselines.py` to compare both models on identical test splits and measure single-window inference latency against sensor arrival intervals ($\Delta t = 1,000\text{ ms}$ for SWaT; $\Delta t = 60,000\text{ ms}$ for SMD).

### 6.1 Evaluate on SWaT

```powershell
python compare_baselines.py `
  --dataset SWAT `
  --train processed/swat/train_normal.npz `
  --test processed/swat/test.npz `
  --latent-ckpt runs/swat/swat_ngls_inspired.pt `
  --label-ckpt runs/swat/swat_label_diff.pt `
  --out-csv runs/swat/comparison_swat.csv
```

### 6.2 Evaluate Across All 28 SMD Machines

```powershell
Get-ChildItem processed\smd -Directory -Filter "machine-*" | ForEach-Object {
    $m = $_.Name
    $lat = "runs\smd\${m}_ngls_inspired.pt"
    $lbl = "runs\smd\${m}_label_diff.pt"
    if ((Test-Path $lat) -and (Test-Path $lbl)) {
        python compare_baselines.py `
          --dataset SMD `
          --train "processed\smd\${m}\train_normal.npz" `
          --test "processed\smd\${m}\test.npz" `
          --latent-ckpt $lat `
          --label-ckpt $lbl `
          --out-csv "runs\smd\comparison_${m}.csv"
    }
}
```

### 6.3 Aggregate 28-Machine SMD Summary Metrics

```powershell
python -c "import glob, pandas as pd; files=sorted(glob.glob('runs/smd/comparison_machine-*.csv')); df=pd.concat([pd.read_csv(f) for f in files], ignore_index=True); df.to_csv('runs/smd/comparison_smd_all.csv', index=False); print(df.groupby('method')[['precision','recall','f1','roc_auc','pr_auc','inference_ms']].mean())"
```

## 7. Step 5: Window-Level Novelty Trace Visualization

Generate diagnostic trace figures comparing normal windows against anomalous windows using regime consensus mismatch and multi-seed variance:

$$S(\mathbf{X}) = \frac{1}{M}\sum_{m=1}^M \mathrm{JSD}(\hat{\mathbf{y}}_0^{(m)} \parallel \mathbf{y}_{\text{ref}}) + \lambda \sum_{k=1}^K \mathrm{Var}_m(\hat{y}_{0,k}^{(m)})$$

### 7.1 SWaT Novelty Trace

```powershell
python trace_windows.py `
  --ckpt runs/swat/swat_label_diff.pt `
  --test processed/swat/test.npz `
  --out-fig runs/swat/novelty_trace_swat.png
```

### 7.2 SMD Novelty Traces (All 28 Machines)

```powershell
Get-ChildItem processed\smd -Directory -Filter "machine-*" | ForEach-Object {
    $m = $_.Name
    $lbl = "runs\smd\${m}_label_diff.pt"
    if (Test-Path $lbl) {
        python trace_windows.py `
          --ckpt $lbl `
          --test "processed\smd\${m}\test.npz" `
          --out-fig "runs\smd\novelty_trace_${m}.png"
    }
}
```

## 8. Benchmark Results Summary

| Dataset & Method | Precision | Recall | F1 | ROC-AUC | PR-AUC | Latency | Speedup vs. $\Delta t$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SWaT:** Latent-Space Diffusion (NGLS) | `0.1935` | `0.7677` | `0.3091` | `0.8419` | `0.7207` | `3.12 ms` | `321×` ($\Delta t = 1,000\text{ ms}$) |
| **SWaT:** One-Class Label-Space Diffusion | `0.0578` | `0.6759` | `0.1066` | `0.7236` | `0.1319` | `5.55 ms` | `180×` ($\Delta t = 1,000\text{ ms}$) |
| **SMD (28m Mean):** Latent-Space Diffusion | `0.1631` | `0.6705` | `0.1967` | `0.8731` | `0.3885` | `4.40 ms` | `15,144×` ($\Delta t = 60,000\text{ ms}$) |
| **SMD (28m Mean):** One-Class Label-Space | `0.1448` | `0.5532` | `0.1701` | `0.7743` | `0.2683` | `7.17 ms` | `9,264×` ($\Delta t = 60,000\text{ ms}$) |

## References

1. Chaofan Li et al., *"Normal-Gathering Latent-Space Diffusion for Time Series Anomaly Detection,"* IEEE Transactions on Industrial Informatics, 2024.
2. Yuhao Chen et al., *"GS: Generative Segmentation via Label Diffusion,"* arXiv:2508.20020, 2025.
3. Yuxin Chen et al., *"ImDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection,"* PVLDB, 2023.
4. Jiaming Song, Chenlin Meng, and Stefano Ermon, *"Denoising Diffusion Implicit Models,"* ICLR, 2021.
5. Aditya P. Mathur and Nils Ole Tippenhauer, *"SWaT: A Water Treatment Testbed for Research and Training on ICS Security,"* CySWater, 2016.