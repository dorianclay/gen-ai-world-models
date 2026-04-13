"""
Train a trajectory diffusion model on antmaze offline data.

Usage:
    python train_antmaze.py --dataset antmaze-umaze-v2
    python train_antmaze.py --dataset_path /path/to/dataset.hdf5

The dataset is downloaded automatically to ~/.datasets/antmaze/ if a name is given
and no local path is provided.
"""
import argparse
import os

import torch
import wandb

from temporal_unet import TemporalUnet
from trajectory_diffusion import GaussianDiffusion
from antmaze_dataset import AntmazeDataset
from trajectory_trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train diffuser on antmaze')

    # dataset
    parser.add_argument('--dataset', default='antmaze-umaze-v2',
                        help='Dataset name (used for auto-download)')
    parser.add_argument('--dataset_path', default=None,
                        help='Local HDF5 path (overrides --dataset auto-download)')
    parser.add_argument('--cache_dir', default=None,
                        help='Directory to cache downloaded datasets')

    # model
    parser.add_argument('--horizon', default=128, type=int,
                        help='Trajectory window length')
    parser.add_argument('--n_diffusion_steps', default=20, type=int)
    parser.add_argument('--dim_mults', nargs='+', default=[1, 2, 4, 8], type=int)
    parser.add_argument('--attention', action='store_true',
                        help='Enable linear attention in TemporalUnet')
    parser.add_argument('--predict_epsilon', action='store_true',
                        help='Predict noise (epsilon) instead of x0')

    # training
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--gradient_accumulate_every', default=2, type=int)
    parser.add_argument('--ema_decay', default=0.995, type=float)
    parser.add_argument('--n_train_steps', default=1_000_000, type=int)
    parser.add_argument('--n_steps_per_epoch', default=10_000, type=int)
    parser.add_argument('--action_weight', default=10.0, type=float)

    # checkpointing / logging
    parser.add_argument('--save_freq', default=20_000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--results_folder', default='./logs/antmaze')
    parser.add_argument('--load_step', default=None, type=int,
                        help='Resume from this checkpoint step')

    # early stopping
    parser.add_argument('--patience', default=5, type=int,
                        help='Stop after this many epochs without improvement (0 to disable)')
    parser.add_argument('--min_delta', default=1e-4, type=float,
                        help='Minimum loss improvement to count as progress')

    # misc
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--wandb_project', default='diffuser-antmaze')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_src = args.dataset_path or args.dataset
    dataset = AntmazeDataset(
        dataset_src,
        horizon=args.horizon,
        cache_dir=args.cache_dir,
    )
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = obs_dim + act_dim
    print(f'obs_dim={obs_dim}  action_dim={act_dim}  transition_dim={transition_dim}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TemporalUnet(
        horizon=args.horizon,
        transition_dim=transition_dim,
        cond_dim=obs_dim,
        dim=32,
        dim_mults=tuple(args.dim_mults),
        attention=args.attention,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=args.horizon,
        observation_dim=obs_dim,
        action_dim=act_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type='l2',
        clip_denoised=False,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_discount=1.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    os.makedirs(args.results_folder, exist_ok=True)
    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        ema_decay=args.ema_decay,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        label_freq=args.n_steps_per_epoch,
        results_folder=args.results_folder,
        device=device,
    )

    if args.load_step is not None:
        trainer.load(args.load_step)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    wandb.init(
        project=args.wandb_project,
        name=os.path.basename(args.results_folder),
        config=vars(args),
        reinit=True,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    n_epochs = args.n_train_steps // args.n_steps_per_epoch
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch} / {n_epochs} | {args.results_folder}')
        mean_loss = trainer.train(n_train_steps=args.n_steps_per_epoch)

        if args.patience > 0:
            if mean_loss < best_loss - args.min_delta:
                best_loss = mean_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f'No improvement for {epochs_without_improvement}/{args.patience} epochs '
                      f'(best={best_loss:.4f}, current={mean_loss:.4f})')
                if epochs_without_improvement >= args.patience:
                    print(f'Early stopping at epoch {epoch}.')
                    wandb.log({'early_stop_epoch': epoch})
                    break

    wandb.finish()


if __name__ == '__main__':
    main()
