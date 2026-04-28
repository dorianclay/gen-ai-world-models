#!/bin/bash
#SBATCH --job-name=ft-dfzn-cont
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
CHECKPTS=("models/antmaze-dfzn-um-JzsoFayH-1820000.pt" "models/antmaze-dfzn-div-4P2d3MVU-1860000.pt")

DSET=${DSETS[$SLURM_ARRAY_TASK_ID]}
CHECKPT=${CHECKPTS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: Starting diffusion finetuning on $DSET"
echo "Job ID: $SLURM_JOB_ID"

srun uv run python experiments/diffusion/train_antmaze.py \
  --dataset "$DSET" \
  --online \
  --load_checkpoint "$CHECKPT" \
  --n_warmup_episodes 500 \
  --n_collect_episodes 200 \
  --collect_freq 5 \
  --eval_freq 10 \
  --n_eval_episodes 30 \
  --n_train_steps 4_000_000
