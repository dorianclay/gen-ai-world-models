#!/bin/bash
#SBATCH --job-name=online
#SBATCH --partition=phil
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/export/doriancl/code/gen-ai-world-models/logs/%x-%A_%a.out

#SBATCH --mail-user=doriancl@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

cd /home/export/doriancl/code/gen-ai-world-models || exit 1

DSETS=("antmaze-umaze-v2" "antmaze-umaze-diverse-v2")

DSET=${DSETS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: Starting diffusion finetuning on $DSET"
echo "Job ID: $SLURM_JOB_ID"

srun uv run python experiments/diffusion/train_antmaze.py \
  --dataset "$DSET" \
  --online \
  --n_warmup_episodes 500 \
  --n_collect_episodes 200 \
  --collect_freq 5 \
  --eval_freq 10
