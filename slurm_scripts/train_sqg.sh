#!/bin/bash
#
# Submit with: sbatch submit_generate.sbatch
#

#SBATCH --job-name=train_sqg        # Job name
#SBATCH --time=2-00:00:00              # Walltime
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --mail-user=sotakao@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL   # Notify at start/end/fail

# --- Choose partition / GPUs as needed ---
#SBATCH --partition=gpu              # Use GPU partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
# If you want CPU only, comment the two lines above and uncomment this:
# #SBATCH --partition=cpu

# --- Optional: log files ---
#SBATCH -o slurm/%x-%j.out           # STDOUT
#SBATCH -e slurm/%x-%j.err           # STDERR

# --- Environment setup ---
source ~/.bashrc
cd /resnick/groups/astuart/sotakao/score-based-ensemble-filter/sda/experiments/sqg
conda activate sda

# --- Run program ---
python train.py --data_dir /central/scratch/sotakao/sqg_train_data \
                --hrly_freq 12
