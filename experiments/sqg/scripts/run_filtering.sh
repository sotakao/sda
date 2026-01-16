#!/bin/sh
#SBATCH -t 02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --gpus=1 -C "thin"

# -J SDA_X

CKPT_PATH="/proj/berzelius-2022-164/weather/SQG/models/sda/sqg_3hrly_latest.pt"
EXPERIMENT="A2"
N_ENS="20"
CORRECTIONS="1"
STEPS="100"
GUIDANCE_METHOD="MMPS"
GUIDANCE_STRENGTH="1.0"
DATA_INDEX="0"
START_TIME="5"

# Parse args passed via sbatch .../run_filtering.sh --data_index X ...
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt_path) CKPT_PATH="$2"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --n_ens) N_ENS="$2"; shift 2 ;;
    --corrections) CORRECTIONS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --guidance_method) GUIDANCE_METHOD="$2"; shift 2 ;;
    --guidance_strength) GUIDANCE_STRENGTH="$2"; shift 2 ;;
    --data_index) DATA_INDEX="$2"; shift 2 ;;
    --start_time) START_TIME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" ; shift ;;
  esac
done

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate sda

cd /proj/berzelius-2022-164/users/x_erila/sda/experiments/sqg

python filtering.py \
    --ckpt_path "${CKPT_PATH}" \
    --experiment "${EXPERIMENT}" \
    --n_ens "${N_ENS}" \
    --corrections "${CORRECTIONS}" \
    --steps "${STEPS}" \
    --guidance_method "${GUIDANCE_METHOD}" \
    --guidance_strength "${GUIDANCE_STRENGTH}" \
    --data_index "${DATA_INDEX}" \
    --start_time "${START_TIME}"