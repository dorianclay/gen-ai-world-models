#!/bin/bash
#SBATCH --job-name=antmaze-dfzn
#SBATCH --partition=phil
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/export/doriancl/code/gen-ai-world-models/logs/%x-%j.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

cd /home/export/doriancl/code/gen-ai-world-models || exit 1

echo "Starting diffusion baseline on AntMaze_UMaze-v5"
echo "Job ID: $SLURM_JOB_ID"

srun uv run python experiments/diffusion/train_antmaze.py \
    --dataset antmaze-umaze-v2 \
    --wandb_project gen-ai-world-models
