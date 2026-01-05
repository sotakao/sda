python filtering.py --data_dir /resnick/groups/astuart/sotakao/score-based-ensemble-filter/FlowDAS/experiments/weather_forecasting/data/sevir_lr \
                    --ckpt_path /resnick/groups/astuart/sotakao/score-based-ensemble-filter/sda/experiments/sevir/runs_sevir/sevirlr_window5_epochs500/checkpoints/latest_aper.pt \
                    --obs_pct 0.1 \
                    --obs_sigma 0.001 \
                    --init_sigma 0.001 \
                    --n_ens 4 \
                    --corrections 2 \
                    --steps 500 \
                    --guidance_method MMPS \
                    --guidance_strength 1.0 
                    