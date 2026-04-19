"""
Train AntMaze Diffuser with a frozen prior and 3M-style progress-guided online collection.

This script is designed for the updated `online` branch of:
  dorianclay/gen-ai-world-models

Place this file in `experiments/diffusion/` or run it from that directory so the
same-folder imports resolve.

What it does:
- Loads a pretrained Diffuser checkpoint twice:
  - frozen prior diffusion model
  - online diffusion model owned by the repo Trainer
- Uses the repo's online fine-tuning pipeline:
  - OnlineAntmazeDataset
  - dataset.refresh(...)
  - trainer.reset_dataloader()
- Replaces plain online rollout collection with 3M-style trajectory ranking:
  - sample multiple candidate trajectories from the online model
  - compute task score from near-final predicted xy
  - compute intrinsic score from denoising disagreement with the frozen prior
  - choose the highest-scoring candidate chunk
- Continues training the online model on refreshed online data

Notes:
- This is a diffusion-native 3M adaptation. The intrinsic reward is based on
  denoising disagreement on the SAME noisy conditioned trajectory, not on a
  one-step transition model.
- This stays aligned with the repo's updated online training path rather than
  maintaining a separate ad hoc update loop.

debug with (took to long):
uv run python experiments\diffusion\antmaze_3m_progress.py --dataset antmaze-umaze-v2 --load_checkpoint models\antmaze-Hn5Trrg9-980000.pt --n_warmup_episodes 10 --n_collect_episodes 5 --collect_freq 1 --eval_freq 1 --samples_per_plan 4 --n_train_steps 20 --n_steps_per_epoch 10 --device cpu

debug with (new):
uv run python experiments\diffusion\antmaze_3m_progress.py --dataset antmaze-umaze-v2 --load_checkpoint models\antmaze-Hn5Trrg9-980000.pt --n_warmup_episodes 5 --n_collect_episodes 1 --collect_freq 1 --eval_freq 0 --samples_per_plan 2 --replan_every 20 --n_train_steps 10 --n_steps_per_epoch 5 --device cpu
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
import shortuuid

from temporal_unet import TemporalUnet
from trajectory_diffusion import GaussianDiffusion
from antmaze_dataset import OnlineAntmazeDataset, DATASET_OBS_DIM
from trajectory_trainer import Trainer
from evaluate import evaluate, dataset_to_env_id

import gymnasium as gym
import gymnasium_robotics  # noqa: F401


DATASET_TO_NAME = {
    'antmaze-umaze-v2': 'U-Maze',
    'antmaze-umaze-diverse-v2': 'U-Maze Diverse',
    'antmaze-big-diverse-v2': 'Big',
    'antmaze-hardest-diverse-v2': 'Hardest',
}


def parse_args(run_id: str):
    parser = argparse.ArgumentParser(description='Train 3M-style Diffuser on AntMaze')

    # dataset / env
    parser.add_argument('--dataset', default='antmaze-umaze-v2')
    parser.add_argument('--env_id', default=None)
    parser.add_argument('--observation_dim', default=None, type=int)
    parser.add_argument('--horizon', default=128, type=int)
    parser.add_argument('--max_path_length', default=700, type=int)

    # online collection / refresh
    parser.add_argument('--n_warmup_episodes', default=1000, type=int)
    parser.add_argument('--n_collect_episodes', default=200, type=int)
    parser.add_argument('--collect_freq', default=5, type=int)
    parser.add_argument('--replan_every', default=5, type=int)
    parser.add_argument('--samples_per_plan', default=8, type=int)

    # model
    parser.add_argument('--n_diffusion_steps', default=20, type=int)
    parser.add_argument('--dim_mults', nargs='+', default=[1, 2, 4, 8], type=int)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--predict_epsilon', action='store_true')
    parser.add_argument('--action_weight', default=10.0, type=float)

    # 3M scoring
    parser.add_argument('--goal_scale', default=1.0, type=float)
    parser.add_argument('--intrinsic_scale', default=1.0, type=float)
    parser.add_argument('--ema_alpha', default=0.10, type=float)
    parser.add_argument('--positive_part', action='store_true')

    # training
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=2e-4, type=float)
    parser.add_argument('--gradient_accumulate_every', default=2, type=int)
    parser.add_argument('--ema_decay', default=0.995, type=float)
    parser.add_argument('--n_train_steps', default=1_000_000, type=int)
    parser.add_argument('--n_steps_per_epoch', default=10_000, type=int)

    # checkpointing / logging
    parser.add_argument('--save_freq', default=20_000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--results_folder', default=f'./logs/antmaze-3m-{run_id}')
    parser.add_argument('--load_checkpoint', required=True, type=str,
                        help='Checkpoint path used for both prior init and online init')

    # evaluation
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--eval_seeds', nargs='+', default=[0, 1, 2], type=int)
    parser.add_argument('--n_eval_episodes', default=100, type=int)

    # early stopping
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--min_delta', default=1e-4, type=float)
    parser.add_argument('--loss_ema_alpha', default=0.1, type=float)

    # misc
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--wandb_project', default='gen-ai-world-models')
    parser.add_argument('--id', type=str, default=run_id)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_goal_obs(desired_goal: np.ndarray, obs_normalizer) -> np.ndarray:
    goal = obs_normalizer.mean.copy()
    goal[:2] = desired_goal
    return goal


def _extract_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        raise ValueError('Checkpoint must deserialize to a dict/state_dict-like object.')

    if 'ema' in ckpt_obj and isinstance(ckpt_obj['ema'], dict):
        return ckpt_obj['ema']

    for key in ['state_dict', 'model_state_dict', 'diffusion', 'model']:
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]

    if any(k.startswith('model.') or k.startswith('betas') for k in ckpt_obj.keys()):
        return ckpt_obj

    raise ValueError(f'Unrecognized checkpoint format. Top-level keys: {list(ckpt_obj.keys())[:20]}')


def load_diffusion_weights(diffusion: GaussianDiffusion, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[warn] missing keys from checkpoint: {missing[:10]}')
    if unexpected:
        print(f'[warn] unexpected keys in checkpoint: {unexpected[:10]}')


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_diffusion(obs_dim: int, act_dim: int, args, device: str) -> GaussianDiffusion:
    transition_dim = obs_dim + act_dim

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
    return diffusion


# ---------------------------------------------------------------------------
# 3M-guided collection
# ---------------------------------------------------------------------------

def make_conditioning(obs_dict: Dict[str, np.ndarray], dataset, horizon: int, device: str):
    body_dim = dataset.observation_dim - 2
    current_full = np.concatenate(
        [obs_dict['achieved_goal'], obs_dict['observation'][:body_dim]],
        axis=0,
    ).astype(np.float32)
    current_norm = dataset.obs_normalizer.normalize(current_full).astype(np.float32)

    goal_full = _make_goal_obs(obs_dict['desired_goal'], dataset.obs_normalizer)
    goal_norm = dataset.obs_normalizer.normalize(goal_full).astype(np.float32)

    return {
        0: torch.tensor(current_norm, device=device).unsqueeze(0),
        horizon - 1: torch.tensor(goal_norm, device=device).unsqueeze(0),
    }


def tile_conditions(cond: Dict[int, torch.Tensor], batch_size: int):
    return {k: v.repeat(batch_size, 1) for k, v in cond.items()}


@torch.no_grad()
def denoising_mismatch_score(
    clean_traj_np: np.ndarray,
    cond_single: Dict[int, torch.Tensor],
    online_diffusion: GaussianDiffusion,
    prior_diffusion: GaussianDiffusion,
    device: str,
) -> float:
    clean_traj = torch.tensor(clean_traj_np, dtype=torch.float32, device=device).unsqueeze(0)
    t = torch.randint(0, online_diffusion.n_timesteps, (1,), device=device).long()
    noise = torch.randn_like(clean_traj)
    noisy_traj = online_diffusion.q_sample(clean_traj, t, noise=noise)

    online_pred = online_diffusion.model(noisy_traj, cond_single, t)
    prior_pred = prior_diffusion.model(noisy_traj, cond_single, t)
    return float(torch.mean((online_pred - prior_pred) ** 2).item())


def goal_score_from_traj(
    traj_norm: np.ndarray,
    obs_normalizer,
    act_dim: int,
    desired_goal_xy: np.ndarray,
) -> float:
    # Avoid scoring the clamped endpoint itself.
    idx = max(len(traj_norm) - 2, 0)
    pred_obs_norm = traj_norm[idx, act_dim:]
    pred_obs = obs_normalizer.unnormalize(pred_obs_norm[None])[0]
    pred_xy = pred_obs[:2]
    dist = np.linalg.norm(pred_xy - desired_goal_xy)
    return -float(dist)


@torch.no_grad()
def plan_chunk_3m(
    online_diffusion: GaussianDiffusion,
    prior_diffusion: GaussianDiffusion,
    dataset,
    obs: Dict[str, np.ndarray],
    args,
    device: str,
    mismatch_ema: Optional[float],
):
    base_cond = make_conditioning(obs, dataset, online_diffusion.horizon, device)
    cond = tile_conditions(base_cond, args.samples_per_plan)

    online_sample = online_diffusion.conditional_sample(cond, verbose=False)
    online_trajs = online_sample.trajectories.detach().cpu().numpy()

    desired_goal_xy = obs['desired_goal']
    scores: List[float] = []
    raw_mismatches: List[float] = []
    intrinsic_scores: List[float] = []
    goal_scores: List[float] = []

    for i in range(args.samples_per_plan):
        online_traj = online_trajs[i]
        cond_single = {k: v[i:i + 1] for k, v in cond.items()}

        g_score = goal_score_from_traj(
            online_traj,
            dataset.obs_normalizer,
            dataset.action_dim,
            desired_goal_xy,
        )
        raw_mismatch = denoising_mismatch_score(
            clean_traj_np=online_traj,
            cond_single=cond_single,
            online_diffusion=online_diffusion,
            prior_diffusion=prior_diffusion,
            device=device,
        )

        if mismatch_ema is None:
            intrinsic = 0.0
        else:
            intrinsic = raw_mismatch - mismatch_ema
            if args.positive_part:
                intrinsic = max(intrinsic, 0.0)

        score = args.goal_scale * g_score + args.intrinsic_scale * intrinsic

        scores.append(score)
        raw_mismatches.append(raw_mismatch)
        intrinsic_scores.append(intrinsic)
        goal_scores.append(g_score)

    best_idx = int(np.argmax(scores))
    best_traj = online_trajs[best_idx]
    acts_norm = best_traj[: args.replan_every, : dataset.action_dim]
    actions = dataset.act_normalizer.unnormalize(acts_norm)

    metrics = {
        'raw_mismatch': raw_mismatches[best_idx],
        'intrinsic': intrinsic_scores[best_idx],
        'goal_score': goal_scores[best_idx],
        'planner_score': scores[best_idx],
    }
    return actions, metrics


@torch.no_grad()
def collect_episodes_3m(
    env_id: str,
    n_episodes: int,
    online_diffusion: GaussianDiffusion,
    prior_diffusion: GaussianDiffusion,
    dataset,
    args,
    device: str,
    seed: Optional[int] = None,
):
    online_diffusion.eval()
    prior_diffusion.eval()

    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    episodes = []
    log_rows = []

    for ep_idx in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        obs_dict, _ = env.reset(seed=ep_seed if ep_idx == 0 else None)

        ep_obs, ep_acts = [], []
        action_buffer = []
        mismatch_ema = None

        for t in range(args.max_path_length):
            body = obs_dict['observation'][: dataset.observation_dim - 2]
            full_obs = np.concatenate([obs_dict['achieved_goal'], body]).astype(np.float32)

            if t % 50 == 0:
                print(f"[collect] episode={ep_idx} step={t}")

            if not action_buffer:
                action_chunk, metrics = plan_chunk_3m(
                    online_diffusion=online_diffusion,
                    prior_diffusion=prior_diffusion,
                    dataset=dataset,
                    obs=obs_dict,
                    args=args,
                    device=device,
                    mismatch_ema=mismatch_ema,
                )
                if mismatch_ema is None:
                    mismatch_ema = metrics['raw_mismatch']
                else:
                    mismatch_ema = (
                        args.ema_alpha * metrics['raw_mismatch']
                        + (1.0 - args.ema_alpha) * mismatch_ema
                    )
                metrics['mismatch_ema'] = float(mismatch_ema)
                log_rows.append(metrics)
                action_buffer = list(action_chunk)

            action = np.clip(
                action_buffer.pop(0),
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)

            ep_obs.append(full_obs)
            ep_acts.append(action)
            obs_dict, _, terminated, truncated, info = env.step(action)
            if terminated or truncated or info.get('success', False):
                break

        if ep_obs:
            episodes.append({
                'observations': np.stack(ep_obs),
                'actions': np.stack(ep_acts),
            })

    env.close()
    online_diffusion.train()

    summary = {
        'collect/n_episodes': len(episodes),
    }
    if log_rows:
        for key in ['raw_mismatch', 'intrinsic', 'goal_score', 'planner_score', 'mismatch_ema']:
            vals = [row[key] for row in log_rows if key in row]
            if vals:
                summary[f'collect/{key}'] = float(np.mean(vals))
    return episodes, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_id = shortuuid.ShortUUID().random(length=8)
    args = parse_args(run_id)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else 'cpu'
    env_id = args.env_id or dataset_to_env_id(args.dataset)
    obs_dim_override = args.observation_dim or DATASET_OBS_DIM.get(args.dataset)

    dataset = OnlineAntmazeDataset(
        env_id,
        horizon=args.horizon,
        n_warmup_episodes=args.n_warmup_episodes,
        max_path_length=args.max_path_length,
        observation_dim=obs_dim_override,
        seed=args.seed,
    )

    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = obs_dim + act_dim
    print(f'obs_dim={obs_dim} action_dim={act_dim} transition_dim={transition_dim}')

    # Online model + trainer
    diffusion = build_diffusion(obs_dim, act_dim, args, device)
    n_params = sum(p.numel() for p in diffusion.model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

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
    trainer.load_from_path(args.load_checkpoint)

    # Frozen prior diffusion from the same checkpoint
    prior_diffusion = build_diffusion(obs_dim, act_dim, args, device)
    load_diffusion_weights(prior_diffusion, args.load_checkpoint, device)
    prior_diffusion.eval()
    for p in prior_diffusion.parameters():
        p.requires_grad = False

    run_name = DATASET_TO_NAME.get(args.dataset, args.dataset) + ' (3M online)'
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        reinit=True,
    )

    n_epochs = args.n_train_steps // args.n_steps_per_epoch
    best_loss = float('inf')
    epochs_without_improvement = 0
    smoothed_loss = None

    for epoch in range(n_epochs):
        print(f'Epoch {epoch} / {n_epochs} | {args.results_folder}')
        mean_loss = trainer.train(n_train_steps=args.n_steps_per_epoch)

        if args.collect_freq > 0 and (epoch + 1) % args.collect_freq == 0:
            print(f'[ Collect 3M ] epoch {epoch} | {args.n_collect_episodes} model episodes from {env_id}')
            new_episodes, collect_summary = collect_episodes_3m(
                env_id=env_id,
                n_episodes=args.n_collect_episodes,
                online_diffusion=trainer.ema_model,
                prior_diffusion=prior_diffusion,
                dataset=dataset,
                args=args,
                device=device,
                seed=args.seed,
            )
            dataset.refresh(new_episodes)
            trainer.reset_dataloader()
            wandb.log(collect_summary, step=trainer.step)

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            print(f'[ Eval ] epoch {epoch} | {env_id}')
            results = evaluate(
                trainer.ema_model,
                dataset,
                env_id,
                n_episodes=args.n_eval_episodes,
                replan_every=args.replan_every,
                seeds=args.eval_seeds,
                device=device,
            )
            wandb.log(
                {
                    'eval/normalized_score': results['normalized_score'],
                    **{f'eval/normalized_score_seed{s}': sc for s, sc in zip(args.eval_seeds, results['seed_scores'])},
                },
                step=trainer.step,
            )
            print(f"[ Eval ] normalized_score={results['normalized_score']:.1f}")

        if args.patience > 0:
            if smoothed_loss is None:
                smoothed_loss = mean_loss
            else:
                smoothed_loss = (
                    args.loss_ema_alpha * mean_loss
                    + (1 - args.loss_ema_alpha) * smoothed_loss
                )

            wandb.log({'loss_smoothed': smoothed_loss}, step=trainer.step)
            if smoothed_loss < best_loss - args.min_delta:
                best_loss = smoothed_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(
                    f'No improvement for {epochs_without_improvement}/{args.patience} epochs '
                    f'(best={best_loss:.4f}, smoothed={smoothed_loss:.4f})'
                )
                if epochs_without_improvement >= args.patience:
                    print(f'Early stopping at epoch {epoch}.')
                    wandb.log({'early_stop_epoch': epoch})
                    break

    trainer.save(trainer.step)
    wandb.finish()


if __name__ == '__main__':
    main()
