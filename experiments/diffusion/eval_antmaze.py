"""
Standalone evaluation script for trained AntMaze diffusion models.

Usage:
    python eval_antmaze.py --checkpoint logs/antmaze/state_500000.pt
    python eval_antmaze.py --checkpoint logs/antmaze/state_500000.pt \\
        --dataset antmaze-medium-play-v2 --replan_every 10 --seeds 0 1 2
"""
import argparse
import os

import numpy as np
import torch
import wandb

from temporal_unet import TemporalUnet
from trajectory_diffusion import GaussianDiffusion
from antmaze_dataset import AntmazeDataset
from evaluate import evaluate, dataset_to_env_id


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate diffusion model on AntMaze')

    # checkpoint
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pt checkpoint (from Trainer.save)')

    # dataset
    parser.add_argument('--dataset', default='antmaze-umaze-v2',
                        help='Dataset name — used for normalizers and env inference')
    parser.add_argument('--dataset_path', default=None,
                        help='Local HDF5 path (overrides --dataset auto-download)')
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--env_id', default=None,
                        help='Gymnasium env id (inferred from --dataset if not set)')

    # model architecture — must match the checkpoint
    parser.add_argument('--horizon', default=128, type=int)
    parser.add_argument('--n_diffusion_steps', default=20, type=int)
    parser.add_argument('--dim_mults', nargs='+', default=[1, 2, 4, 8], type=int)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--predict_epsilon', action='store_true')
    parser.add_argument('--action_weight', default=10.0, type=float)

    # evaluation
    parser.add_argument('--n_episodes', default=100, type=int,
                        help='Episodes per seed (D4RL standard: 100)')
    parser.add_argument('--replan_every', default=5, type=int,
                        help='Replan after this many environment steps')
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2], type=int)

    # misc
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wandb_project', default='diffuser-antmaze')
    parser.add_argument('--wandb_run_name', default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------
    # Dataset (needed for normalizers — no episodes are sampled)
    # ------------------------------------------------------------------
    dataset_src = args.dataset_path or args.dataset
    dataset = AntmazeDataset(dataset_src, horizon=args.horizon, cache_dir=args.cache_dir)
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    unet = TemporalUnet(
        horizon=args.horizon,
        transition_dim=obs_dim + act_dim,
        cond_dim=obs_dim,
        dim=32,
        dim_mults=tuple(args.dim_mults),
        attention=args.attention,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=unet,
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

    # ------------------------------------------------------------------
    # Load checkpoint — use EMA weights for evaluation
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    diffusion.load_state_dict(ckpt['ema'])
    print(f'Loaded EMA weights from {args.checkpoint} (step {ckpt["step"]})')

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    env_id = args.env_id or dataset_to_env_id(args.dataset)
    print(f'Evaluating on {env_id} | {args.n_episodes} episodes × {len(args.seeds)} seeds')

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f'eval-{os.path.basename(args.checkpoint)}',
        config=vars(args),
        reinit=True,
    )

    results = evaluate(
        diffusion, dataset, env_id,
        n_episodes=args.n_episodes,
        replan_every=args.replan_every,
        seeds=args.seeds,
        device=device,
    )

    std = float(np.std(results['seed_scores']))
    print(f'\nNormalized score: {results["normalized_score"]:.1f} ± {std:.1f}')

    wandb.log({
        'normalized_score': results['normalized_score'],
        'normalized_score_std': std,
        **{f'normalized_score_seed{s}': sc
           for s, sc in zip(args.seeds, results['seed_scores'])},
        'step': ckpt['step'],
    })
    wandb.finish()


if __name__ == '__main__':
    main()
