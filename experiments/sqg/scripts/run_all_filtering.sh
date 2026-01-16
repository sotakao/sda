#!/bin/sh

# data_idxs=(0 1 2 3 4 5 6 7 8 9) 
data_idxs=(0) 


for data_idx in "${data_idxs[@]}"; do
    job_name="SDA_${data_idx}"
    sbatch --job-name="$job_name" /proj/berzelius-2022-164/users/x_erila/sda/experiments/sqg/scripts/run_filtering.sh \
        --data_index "${data_idx}"
done