#!/bin/bash
#
# Save as: submit_inference_grid.sbatch
# Submit with: sbatch submit_inference_grid.sbatch
#
# Total runs: 5 (guidance_strength) * 3 (steps) * 3 (corrections) = 45

#SBATCH --job-name=SQG_A4_MMPS_Sweep
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --mail-user=sotakao@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Choose partition / GPUs as needed ---
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
# If CPU only, comment the two lines above and uncomment this:
# #SBATCH --partition=cpu

# --- Job array over all parameter combinations (0..44) ---
#SBATCH --array=0-44%5

# --- Optional: log files ---
#SBATCH -o slurm/%x-%A_%a.out
#SBATCH -e slurm/%x-%A_%a.err

set -eo pipefail
echo "[START] $(date) job=$SLURM_JOB_ID task=${SLURM_ARRAY_TASK_ID:-NA}" 1>&2

# --- Ensure log directory exists ---
mkdir -p slurm

# --- Environment setup ---
source /groups/astuart/sotakao/miniconda3/etc/profile.d/conda.sh
conda activate /groups/astuart/sotakao/miniconda3/envs/sda
cd /resnick/groups/astuart/sotakao/score-based-ensemble-filter/sda/experiments/sqg

# ---------------- Grid definition ----------------
# Indices:
#   a = SLURM_ARRAY_TASK_ID in [0..44]
#   gs_idx   = a % 5
#   steps_idx= (a / 5) % 3
#   corr_idx = (a / (5*3)) % 3

GUIDES=(0.1 0.5 1.0 3.0 5.0)
STEPSS=(100 250 500)
CORRS=(1 2)

a=${SLURM_ARRAY_TASK_ID}

gs_idx=$(( a % 5 ))
steps_idx=$(( (a / 5) % 3 ))
corr_idx=$(( (a / 15) % 2 ))

GUIDE=${GUIDES[$gs_idx]}
STEPS=${STEPSS[$steps_idx]}
CORRECTIONS=${CORRS[$corr_idx]}

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: guidance_strength=${GUIDE}, steps=${STEPS}, corrections=${CORRECTIONS}"


# ---------------- Launch ----------------
# srun python inference2.py \
#     --data_dir /central/scratch/sotakao/sqg_train_data \
#     --train_file sqg_pv_train.h5 \
#     --hrly_freq 3 \
#     --obs_pct 0.25 \
#     --obs_fn square_scaled \
#     --obs_sigma 1.0 \
#     --n_ens 20 \
#     --log_wandb 1 \
#     --guidance_method MMPS \
#     --guidance_strength "${GUIDE}" \
#     --corrections "${CORRECTIONS}" \
#     --steps "${STEPS}" \
#     --initial_condition

srun python smoothing.py\
    --val_file /resnick/groups/astuart/sotakao/score-based-ensemble-filter/EnSFInpainting/data/test/sqg_N64_3hrly_100.nc \
    --ckpt_path /resnick/groups/astuart/sotakao/score-based-ensemble-filter/sda/runs_sqg/mcscore_vpsde_sqg_window_5/checkpoints/sqg_3hrly_latest.pt \
    --obs_pct 0.25 \
    --obs_fn square_scaled \
    --obs_sigma 1.0 \
    --init_sigma 1.0 \
    --n_ens 20 \
    --corrections "${CORRECTIONS}" \
    --steps "${STEPS}" \
    --guidance_method MMPS \
    --guidance_strength "${GUIDE}" \
    --initial_condition

echo "[$(date)] Done."
