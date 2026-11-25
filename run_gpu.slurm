#!/bin/bash -l
#SBATCH --job-name=anomaly_detection
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/cccp/25m2117/root/Diffusion_Model/swat_smd_anomaly_detection_%j.out
#SBATCH --error=/home/cccp/25m2117/root/Diffusion_Model/swat_smd_anomaly_detection_%j.err

set -euo pipefail


export DS_SKIP_CUDA_CHECK=1
ROOT="/home/cccp/25m2117/root/Diffusion_Model/swat_smd_anomaly_detection"


LOG_DIR="${ROOT}/logs/${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/train_stdout.log")
exec 2> >(tee -a "${LOG_DIR}/train_stderr.log" >&2)

############################################
# Environment
############################################

#export PYTHONUNBUFFERED=1
#export HYDRA_FULL_ERROR=1
#export TOKENIZERS_PARALLELISM=false

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

#export VLLM_ATTENTION_BACKEND=XFORMERS

############################################
# Activate Conda
############################################

CONDA_BASE=$(conda info --base 2>/dev/null)

if [ -z "$CONDA_BASE" ]; then
    echo "Unable to locate conda."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate SMD

############################################
# Cleanup
############################################

if command -v ray >/dev/null 2>&1; then
    ray stop --force 2>/dev/null || true
fi



echo "========================================"
echo "Hostname      : $(hostname)"
echo "Date          : $(date)"
echo "Job ID        : ${SLURM_JOB_ID}"

echo "========================================"

nvidia-smi


echo
############################################
# Launch
############################################

cd "${ROOT}"



#python exe_machine.py \
#    --device cuda:0 \
#    --dataset SMD
#python evaluate_machine_window_middle.py --device cuda:0 --dataset SMD
#python train_diffusion_.py \
#  --train processed/swat/train_normal.npz \
#  --test processed/swat/test.npz \
#  --window 100 --stride 10 --epochs 10 \
#  --batch-size 128 --steps 100 --draws 4 \
#  --lr 1e-3 --threshold-percentile 99 --seed 42 \
#  --out runs/diffusion_swat.pt
#python train_ngls_.py \
#  --train processed/swat/train_normal.npz \
#  --test processed/swat/test.npz \
#  --window 100 --stride 10 --ae-epochs 10 --diff-epochs 10 \
#  --batch-size 128 --steps 100 --draws 4 \
#  --lr 1e-3 --threshold-percentile 99 --seed 42 \
#  --score-mode combined \
#  --out runs/ngls_combined_only_swat.pt  
# SWaT
#python train_label_diffusion.py --train processed/swat/train_normal.npz --test processed/swat/test.npz --num-regimes 24 --out runs/label_diff_swat_k24.pt

# All 28 SMD machines
#python train_label_diffusion.py --all-smd --smd-dir processed/smd --results-csv runs/smd/label_diff_results.csv
export MPLBACKEND=Agg

for dir in processed/smd/machine-*; do
    m=$(basename "$dir")
    lbl_ckpt="runs/smd/${m}_label_diff.pt"

    if [[ -f "runs/smd/${m}.pt" ]]; then
        lat_ckpt="runs/smd/${m}.pt"
    elif [[ -f "runs/smd/${m}_ngls_inspired.pt" ]]; then
        lat_ckpt="runs/smd/${m}_ngls_inspired.pt"
    else
        lat_ckpt=""
    fi

    if [[ -f "$lbl_ckpt" ]]; then
        echo "=== Tracing $m ==="
        python trace_windows.py \
            --ckpt "$lbl_ckpt" \
            --test "${dir}/test.npz" \
            --out-fig "runs/smd/novelty_trace_${m}.png"
    fi

    if [[ -f "$lbl_ckpt" && -n "$lat_ckpt" ]]; then
        echo "=== Comparing $m (latent: $lat_ckpt) ==="
        python compare_baselines.py \
            --dataset SMD \
            --train "${dir}/train_normal.npz" \
            --test "${dir}/test.npz" \
            --latent-ckpt "$lat_ckpt" \
            --label-ckpt "$lbl_ckpt" \
            --out-csv "runs/smd/comparison_${m}.csv"
    else
        echo "[Skip compare $m] Missing label or latent checkpoint"
    fi
done

python -c '
import glob, pandas as pd
files = sorted(glob.glob("runs/smd/comparison_machine-*.csv"))
if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv("runs/smd/comparison_smd_all.csv", index=False)
    print(df.groupby("method")[["precision","recall","f1","roc_auc","pr_auc","inference_ms"]].mean())
'


#python trace_windows.py --ckpt runs/label_diff_swat.pt --test processed/swat/test.npz --out-fig runs/novelty_trace.png


#python compare_baselines.py --dataset SWaT --train processed/swat/train_normal.npz --test processed/swat/test.npz --latent-ckpt runs/ngls_combined_only_swat.pt --label-ckpt runs/label_diff_swat.pt --out-csv runs/comparison_swat.csv
echo
echo "Training completed at: $(date)"