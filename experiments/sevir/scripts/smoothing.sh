# python smoothing.py --data_dir /resnick/groups/astuart/sotakao/score-based-ensemble-filter/FlowDAS/experiments/weather_forecasting/data/sevir_lr \
#                     --ckpt_path /resnick/groups/astuart/sotakao/score-based-ensemble-filter/sda/experiments/sevir/runs_sevir/sevirlr_window5_epochs500/checkpoints/latest_aper.pt \
#                     --in_len 6 \
#                     --obs_pct 0.1 \
#                     --obs_sigma 0.001 \
#                     --init_sigma 0.001 \
#                     --n_ens 4 \
#                     --corrections 2 \
#                     --steps 500 \
#                     --guidance_method MMPS \
#                     --guidance_strength 1.0 \
#                     --initial_condition

DATA_DIR="/proj/berzelius-2022-164/weather/SEVIR/sevir_lr"
CKPT_PATH="/proj/berzelius-2022-164/weather/SEVIR/sevir_latest.pt"
OBS_PCT="0.1"
OBS_SIGMA="0.001"
INIT_SIGMA="0.001"
N_ENS="4"
CORRECTIONS="2"
STEPS="10"
GUIDANCE_METHOD="MMPS"
GUIDANCE_STRENGTH="1.0"
DATA_INDEX="0"
START_TIME="0"

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate sda

cd /proj/berzelius-2022-164/users/x_erila/sda/experiments/sevir

python smoothing.py \
    --data_dir "${DATA_DIR}" \
    --ckpt_path "${CKPT_PATH}" \
    --obs_pct "${OBS_PCT}" \
    --obs_sigma "${OBS_SIGMA}" \
    --init_sigma "${INIT_SIGMA}" \
    --n_ens "${N_ENS}" \
    --corrections "${CORRECTIONS}" \
    --steps "${STEPS}" \
    --guidance_method "${GUIDANCE_METHOD}" \
    --guidance_strength "${GUIDANCE_STRENGTH}" \
    --data_index "${DATA_INDEX}" \
    --start_time "${START_TIME}" \
    --in_len 6 \
    --initial_condition \