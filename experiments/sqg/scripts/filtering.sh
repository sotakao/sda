python filtering.py --val_file /resnick/groups/astuart/sotakao/score-based-ensemble-filter/EnSFInpainting/data/test/sqg_N64_3hrly_100.nc \
                    --ckpt_path /resnick/groups/astuart/sotakao/score-based-ensemble-filter/sda/runs_sqg/mcscore_vpsde_sqg_window_5/checkpoints/sqg_3hrly_latest.pt \
                    --obs_pct 0.05 \
                    --obs_fn linear \
                    --obs_sigma 1.0 \
                    --init_sigma 1.0 \
                    --n_ens 20 \
                    --log_wandb 1 \
                    --corrections 1 \
                    --guidance_method MMPS \
                    --guidance_strength 1.0 \
                    --window 5 \
                    --steps 100
                    