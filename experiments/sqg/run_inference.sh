source ~/.bashrc
conda activate sda
python inference2.py --data_dir /resnick/scratch/sotakao/sqg_train_data \
                    --train_file sqg_pv_train.h5 \
                    --hrly_freq 3 \
                    --obs_pct 0.25 \
                    --obs_fn arctan \
                    --obs_sigma 0.01 \
                    --n_ens 4 \
                    --log_wandb 1 \
                    --corrections 1 \
                    --guidance_method DPS \
                    --guidance_strength 0.1

# Ablation over:
# - obs_pct: 0.05, 0.25
# - obs_fn: linear, arctan15
# - guidance_method: DPS, MMPS
# - guidance_strength: 0.1, 1.0
# - corrections: 0, 1, 2
# total runs: 2 * 2 * 2 * 3 * 3 = 72