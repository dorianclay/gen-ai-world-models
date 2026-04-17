#!/bin/bash
#SBATCH --job-name=antmaze-dfzn
#SBATCH --partition=phil
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/export/doriancl/code/gen-ai-world-models/logs/%x-%A_%a.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

DSETS=("antmaze-umaze-v2" "antmaze-umaze-diverse-v2")

DSET=${DSETS[$SLURM_ARRAY_TASK_ID]}

cd /home/export/doriancl/code/gen-ai-world-models || exit 1

echo "Task $SLURM_ARRAY_TASK_ID: Starting diffusion baseline on $DSET"
echo "Job ID: $SLURM_JOB_ID"

srun uv run python experiments/diffusion/train_antmaze.py \
    --dataset "$DSET" \
    --wandb_project gen-ai-world-models \
    --eval_freq 10 \
    --n_eval_episodes 30
