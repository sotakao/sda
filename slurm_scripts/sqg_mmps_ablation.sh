#!/bin/bash
#SBATCH -J sqg_ablate
#SBATCH --array=0-71
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# #SBATCH --partition=gpu                # ← uncomment & set if you need GPUs
# #SBATCH --gres=gpu:1                   # ← uncomment if you need a GPU
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err
#SBATCH --requeue

set -euo pipefail

### --- ENVIRONMENT ---
source ~/.bashrc
conda activate sda

# Optional: keep WandB happy on clusters
export WANDB__SERVICE_WAIT=300
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

### --- FIXED ARGS (edit as needed) ---
DATA_DIR="/central/scratch/sotakao/sqg_train_data"
TRAIN_FILE="sqg_pv_train.h5"
HRLY_FREQ=3
N_ENS=5
LOG_WANDB=1
GUIDANCE_METHOD=MMPS

### --- SWEEP SPACE ---
# NOTE: To achieve 72 total runs, we use three guidance strengths.
OBS_PCTS=(0.05 0.25)                 # 2
OBS_FNS=(linear arctan15)            # 2
GUIDE_STRENGTHS=(0.01 0.1 0.5 1.0)        # 4  ← if you really want only (0.1, 1.0), change here (will make 48 runs)
CORRECTIONS=(0 1 2)                  # 3

N1=${#OBS_PCTS[@]}           # 2
N2=${#OBS_FNS[@]}            # 2
N3=${#GUIDE_STRENGTHS[@]}    # 3
N4=${#CORRECTIONS[@]}        # 3

TOTAL=$((N1*N2*N3*N4))
if [[ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL} ]]; then
  echo "Array index ${SLURM_ARRAY_TASK_ID} is out of range (0..$((TOTAL-1)))."
  exit 1
fi

# --- Mixed-radix decode of SLURM_ARRAY_TASK_ID into indices ---
i=${SLURM_ARRAY_TASK_ID}

i0=$(( i % N1 ));         i=$(( i / N1 ))
i1=$(( i % N2 ));         i=$(( i / N2 ))
i2=$(( i % N3 ));         i=$(( i / N3 ))
i3=$(( i % N4 ));         # last one

OBS_PCT=${OBS_PCTS[$i0]}
OBS_FN=${OBS_FNS[$i1]}
GUIDANCE_STRENGTH=${GUIDE_STRENGTHS[$i3]}
CORR=${CORRECTIONS[$i4]}

# Nice, unique run name/ID for logs & wandb grouping
RUN_TAG="pct${OBS_PCT}_fn${OBS_FN}_gm${GUIDANCE_METHOD}_gs${GUIDANCE_STRENGTH}_corr${CORR}"
echo "[$(date)] Starting run ${RUN_TAG} (array id ${SLURM_ARRAY_TASK_ID}/${TOTAL})"

# Optional: per-run output dir (python might also handle logging)
OUTDIR="outputs/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
mkdir -p "${OUTDIR}"

# If you want WandB grouping across the sweep:
export WANDB_RUN_GROUP="sqg_ablate_${SLURM_JOB_ID}"
export WANDB_RUN_NAME="${RUN_TAG}"

# --- Launch ---
srun python inference2.py \
  --data_dir "${DATA_DIR}" \
  --train_file "${TRAIN_FILE}" \
  --hrly_freq "${HRLY_FREQ}" \
  --obs_pct "${OBS_PCT}" \
  --obs_fn "${OBS_FN}" \
  --n_ens "${N_ENS}" \
  --log_wandb "${LOG_WANDB}" \
  --corrections "${CORR}" \
  --guidance_method "${GUIDANCE_METHOD}" \
  --guidance_strength "${GUIDANCE_STRENGTH}" \
  --run_tag "${RUN_TAG}" \
  --out_dir "${OUTDIR}"

echo "[$(date)] Finished run ${RUN_TAG}"
