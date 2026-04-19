"""
Closed-loop AntMaze evaluation with a frozen Diffuser prior and an online Diffuser planner, can lowkey delete.

This stays close to the repo's past implementation

- load TWO diffusion models with identical architecture
- freeze one as the prior
- use the other as the planner
- score online trajectory samples with:
      score = goal_score + lambda * intrinsic_score
  where intrinsic_score is trajectory disagreement with the frozen prior

This is a trajectory-level 3M-style extension, not a one-step transition model implementation.

dummy usage for quick test: 

uv run python experiments\diffusion\eval_mismatch.py --dataset antmaze-umaze-v2 --prior_ckpt models\antmaze-Hn5Trrg9-980000.pt --samples_per_plan 4 --replan_every 5 --n_episodes 1 --max_steps 50 --device cpu
"""

import argparse
import copy
import os
from typing import Dict, Tuple

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch

from temporal_unet import TemporalUnet
from trajectory_diffusion import GaussianDiffusion
from antmaze_dataset import AntmazeDataset
from evaluate import dataset_to_env_id

def parse_args():
    parser = argparse.ArgumentParser(description="AntMaze planning with frozen Diffuser prior")

    # data / env
    parser.add_argument("--dataset", default="antmaze-umaze-v2")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--env_id", default=None)

    # checkpoints
    parser.add_argument("--prior_ckpt", required=True, help="Frozen prior checkpoint")
    parser.add_argument(
        "--online_ckpt",
        default=None,
        help="Online checkpoint. If omitted, initialize from prior_ckpt.",
    )

    # model config: keep aligned with train_antmaze.py
    parser.add_argument("--horizon", default=128, type=int)
    parser.add_argument("--n_diffusion_steps", default=20, type=int)
    parser.add_argument("--dim_mults", nargs="+", default=[1, 2, 4, 8], type=int)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--predict_epsilon", action="store_true")
    parser.add_argument("--action_weight", default=10.0, type=float)

    # planning
    parser.add_argument("--replan_every", default=5, type=int)
    parser.add_argument("--samples_per_plan", default=32, type=int)
    parser.add_argument("--max_steps", default=700, type=int)

    # intrinsic scoring
    parser.add_argument("--intrinsic_scale", default=1.0, type=float)
    parser.add_argument("--goal_scale", default=1.0, type=float)
    parser.add_argument(
        "--mismatch_mode",
        choices=["full_traj", "actions", "obs"],
        default="full_traj",
        help="What part of the sampled trajectory to compare.",
    )
    parser.add_argument(
        "--ema_alpha",
        default=0.10,
        type=float,
        help="EMA smoothing for mismatch baseline",
    )
    parser.add_argument(
        "--positive_part",
        action="store_true",
        help="Use max(mismatch - ema, 0) instead of signed difference",
    )

    # evaluation
    parser.add_argument("--n_episodes", default=50, type=int)
    parser.add_argument("--seeds", nargs="+", default=[0, 1, 2], type=int)

    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _make_goal_obs(desired_goal: np.ndarray, obs_normalizer):
    """
    Matches the repo's evaluate.py:
    set xy dims to desired_goal and body dims to dataset mean.
    """
    goal = obs_normalizer.mean.copy()
    goal[:2] = desired_goal
    return goal


def build_diffusion(dataset, args, device: str):
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
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
        loss_type="l2",
        clip_denoised=False,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_discount=1.0,
    ).to(device)

    return diffusion


def _extract_state_dict(ckpt_obj):
    """
    Robust checkpoint loading:
    - raw state_dict
    - trainer checkpoint with ema/model keys
    - nested diffusion/model dicts
    """
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Checkpoint must deserialize to a dict/state_dict-like object.")

    # Prefer EMA if present, since train_antmaze.py evaluates trainer.ema_model
    if "ema" in ckpt_obj and isinstance(ckpt_obj["ema"], dict):
        return ckpt_obj["ema"]

    # Common alternate names
    for key in ["state_dict", "model_state_dict", "diffusion", "model"]:
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]

    # Raw state_dict case
    if any(k.startswith("model.") or k.startswith("betas") for k in ckpt_obj.keys()):
        return ckpt_obj

    raise ValueError(f"Unrecognized checkpoint format. Top-level keys: {list(ckpt_obj.keys())[:20]}")


def load_diffusion_weights(diffusion: GaussianDiffusion, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[warn] missing keys from checkpoint: {missing[:10]}")
    if unexpected:
        print(f"[warn] unexpected keys in checkpoint: {unexpected[:10]}")


def make_conditioning(obs: Dict[str, np.ndarray], dataset, horizon: int, device: str):
    """
    Matches evaluate.py conditioning logic.
    """
    body_dim = dataset.observation_dim - 2
    current_full = np.concatenate(
        [obs["achieved_goal"], obs["observation"][:body_dim]],
        axis=0,
    )
    current_norm = dataset.obs_normalizer.normalize(current_full).astype(np.float32)

    goal_full = _make_goal_obs(obs["desired_goal"], dataset.obs_normalizer)
    goal_norm = dataset.obs_normalizer.normalize(goal_full).astype(np.float32)

    return {
        0: torch.tensor(current_norm, device=device).unsqueeze(0),
        horizon - 1: torch.tensor(goal_norm, device=device).unsqueeze(0),
    }


def tile_conditions(cond: Dict[int, torch.Tensor], batch_size: int):
    return {k: v.repeat(batch_size, 1) for k, v in cond.items()}


def mismatch_score(
    online_traj: np.ndarray,
    prior_traj: np.ndarray,
    act_dim: int,
    mode: str,
) -> float:
    if mode == "actions":
        online_part = online_traj[:, :act_dim]
        prior_part = prior_traj[:, :act_dim]
    elif mode == "obs":
        online_part = online_traj[:, act_dim:]
        prior_part = prior_traj[:, act_dim:]
    else:
        online_part = online_traj
        prior_part = prior_traj

    return float(np.mean((online_part - prior_part) ** 2))


def goal_score_from_traj(
    traj_norm: np.ndarray,
    obs_normalizer,
    act_dim: int,
    desired_goal_xy: np.ndarray,
) -> float:
    """
    Keep scoring simple and faithful to goal-conditioned AntMaze:
    use final predicted xy distance to desired goal.
    Higher is better.
    """
    final_obs_norm = traj_norm[-1, act_dim:]
    final_obs = obs_normalizer.unnormalize(final_obs_norm[None])[0]
    final_xy = final_obs[:2]
    dist = np.linalg.norm(final_xy - desired_goal_xy)
    return -float(dist)

@torch.no_grad()
def denoising_mismatch_score(
    clean_traj_np: np.ndarray,
    cond_single: Dict[int, torch.Tensor],
    online_diffusion: GaussianDiffusion,
    prior_diffusion: GaussianDiffusion,
    device: str,
) -> float:
    """
    Compare online vs prior denoising predictions on the SAME noisy trajectory
    under the SAME conditioning.
    """
    clean_traj = torch.tensor(
        clean_traj_np, dtype=torch.float32, device=device
    ).unsqueeze(0)

    t = torch.randint(
        low=0,
        high=online_diffusion.n_timesteps,
        size=(1,),
        device=device,
    ).long()

    noise = torch.randn_like(clean_traj)
    noisy_traj = online_diffusion.q_sample(clean_traj, t, noise=noise)

    online_pred = online_diffusion.model(noisy_traj, cond_single, t)
    prior_pred = prior_diffusion.model(noisy_traj, cond_single, t)

    return float(torch.mean((online_pred - prior_pred) ** 2).item())

@torch.no_grad()
def plan_chunk(
    online_diffusion: GaussianDiffusion,
    prior_diffusion: GaussianDiffusion,
    dataset,
    obs: Dict[str, np.ndarray],
    args,
    device: str,
    mismatch_ema: float | None,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Sample multiple online trajectories, score each with:
      goal score + intrinsic score
    where intrinsic score is denoising-prediction disagreement between
    the online and frozen prior diffusion models on the SAME noisy trajectory.
    """
    base_cond = make_conditioning(obs, dataset, online_diffusion.horizon, device)
    cond = tile_conditions(base_cond, args.samples_per_plan)

    # Sample candidate online trajectories only
    online_sample = online_diffusion.conditional_sample(cond, verbose=False)
    online_trajs = online_sample.trajectories.detach().cpu().numpy()

    desired_goal_xy = obs["desired_goal"]
    scores = []
    raw_mismatches = []
    intrinsic_scores = []
    goal_scores = []

    for i in range(args.samples_per_plan):
        
        online_traj = online_trajs[i]

        g_score = goal_score_from_traj(
            online_traj,
            dataset.obs_normalizer,
            dataset.action_dim,
            desired_goal_xy,
        )
        cond_single = {k: v[i:i+1] for k, v in cond.items()}
        
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

        print(f"goal={g_score:.3f}, mismatch={raw_mismatch:.5f}, intrinsic={intrinsic:.5f}")

    best_idx = int(np.argmax(scores))
    best_traj = online_trajs[best_idx]

    acts_norm = best_traj[: args.replan_every, : dataset.action_dim]
    actions = dataset.act_normalizer.unnormalize(acts_norm)

    best_raw_mismatch = raw_mismatches[best_idx]
    best_intrinsic = intrinsic_scores[best_idx]
    best_goal = goal_scores[best_idx]

    return actions, best_raw_mismatch, best_intrinsic, best_goal





def run_episode(
    online_diffusion,
    prior_diffusion,
    dataset,
    env,
    args,
    device: str,
    seed=None,
):
    obs, _ = env.reset(seed=seed)
    action_buffer = []
    mismatch_ema = None

    episode_return_intrinsic = 0.0
    episode_goal_score = 0.0

    for t in range(args.max_steps):
        if not action_buffer:
            actions, raw_mismatch, intrinsic_bonus, goal_score = plan_chunk(
                online_diffusion=online_diffusion,
                prior_diffusion=prior_diffusion,
                dataset=dataset,
                obs=obs,
                args=args,
                device=device,
                mismatch_ema=mismatch_ema,
            )

            # Update EMA after observing chosen plan's mismatch
            if mismatch_ema is None:
                mismatch_ema = raw_mismatch
            else:
                mismatch_ema = args.ema_alpha * raw_mismatch + (1.0 - args.ema_alpha) * mismatch_ema

            episode_return_intrinsic += intrinsic_bonus
            episode_goal_score += goal_score
            action_buffer = list(actions)

        action = np.clip(
            action_buffer.pop(0),
            env.action_space.low,
            env.action_space.high,
        )

        obs, _, terminated, truncated, info = env.step(action)

        if info.get("success", False):
            return {
                "success": True,
                "steps": t + 1,
                "intrinsic_total": episode_return_intrinsic,
                "goal_score_total": episode_goal_score,
            }

        if terminated or truncated:
            break

    return {
        "success": False,
        "steps": args.max_steps,
        "intrinsic_total": episode_return_intrinsic,
        "goal_score_total": episode_goal_score,
    }


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    dataset_src = args.dataset_path or args.dataset
    dataset = AntmazeDataset(
        dataset_src,
        horizon=args.horizon,
        cache_dir=args.cache_dir,
    )

    env_id = args.env_id or dataset_to_env_id(args.dataset)

    # Build two identical diffusion models
    prior_diffusion = build_diffusion(dataset, args, device)
    online_diffusion = build_diffusion(dataset, args, device)

    # Load weights
    load_diffusion_weights(prior_diffusion, args.prior_ckpt, device)
    if args.online_ckpt is None:
        online_diffusion.load_state_dict(copy.deepcopy(prior_diffusion.state_dict()))
    else:
        load_diffusion_weights(online_diffusion, args.online_ckpt, device)

    # Freeze prior
    prior_diffusion.eval()
    for p in prior_diffusion.parameters():
        p.requires_grad = False

    # Planner model in eval mode too, since this script is for closed-loop planning
    online_diffusion.eval()

    all_seed_scores = []

    for seed in args.seeds:
        env = gym.make(env_id)
        successes = 0
        step_counts = []
        intrinsic_totals = []
        goal_totals = []

        for ep in range(args.n_episodes):
            ep_seed = seed if ep == 0 else None
            result = run_episode(
                online_diffusion=online_diffusion,
                prior_diffusion=prior_diffusion,
                dataset=dataset,
                env=env,
                args=args,
                device=device,
                seed=ep_seed,
            )
            successes += int(result["success"])
            step_counts.append(result["steps"])
            intrinsic_totals.append(result["intrinsic_total"])
            goal_totals.append(result["goal_score_total"])

        env.close()

        score = successes / args.n_episodes * 100.0
        all_seed_scores.append(score)

        print(
            f"[seed={seed}] "
            f"successes={successes}/{args.n_episodes} "
            f"score={score:.1f} "
            f"mean_steps={np.mean(step_counts):.1f} "
            f"mean_intrinsic={np.mean(intrinsic_totals):.4f} "
            f"mean_goal_score={np.mean(goal_totals):.4f}"
        )

    print(f"\nnormalized_score={float(np.mean(all_seed_scores)):.2f}")


if __name__ == "__main__":
    main()