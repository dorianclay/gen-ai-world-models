"""
Closed-loop evaluation for trajectory diffusion models on AntMaze.

The diffusion model acts as an MPC planner: every `replan_every` steps it
samples a full trajectory conditioned on (current_obs, goal_obs), then
executes the first `replan_every` actions before replanning.

Normalized score = success_rate * 100, averaged over seeds.
(AntMaze: random ≈ 0, expert ≈ 100)
"""
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics  # noqa: registers AntMaze envs
from exploration_metrics import compute_xy_coverage_metrics_from_episodes


# D4RL dataset name → gymnasium-robotics env id
DATASET_TO_ENV = {
    'antmaze-umaze-v2':         'AntMaze_UMaze-v5',
    'antmaze-umaze-diverse-v2': 'AntMaze_UMaze-v5',
    'antmaze-medium-play-v2':   'AntMaze_Medium_Diverse_GR-v4',
    'antmaze-large-play-v2':    'AntMaze_Large_Diverse_GR-v4',
    'antmaze-big-diverse-v2':   'AntMaze_BigMaze_DG-v5',
    'antmaze-hardest-diverse-v2':'AntMaze_HardestMaze_DG-v5',
}


def dataset_to_env_id(dataset_name):
    if dataset_name in DATASET_TO_ENV:
        return DATASET_TO_ENV[dataset_name]
    raise ValueError(
        f'No known gym env for dataset "{dataset_name}". '
        f'Pass --env_id explicitly. Known: {list(DATASET_TO_ENV)}'
    )


def _make_goal_obs(desired_goal, obs_normalizer):
    """
    Build a full-dim goal observation for diffusion conditioning.

    Sets xy dims to desired_goal; body dims to the dataset mean so that
    after normalization the body part becomes 0 (neutral / average state).
    """
    goal = obs_normalizer.mean.copy()
    goal[:2] = desired_goal
    return goal


def _rollout_episode(diffusion, dataset, env, replan_every, device,
                     max_steps=700, seed=None):
    """Run one episode; returns True on success."""
    obs, _ = env.reset(seed=seed)
    action_buffer = []

    for _ in range(max_steps):
        if not action_buffer:
            # gymnasium-robotics v5 obs['observation'] includes contact forces (105D),
            # but D4RL datasets only contain qpos[2:]+qvel (27D). Take only the
            # matching prefix so shapes align with the normalizer.
            body_dim = dataset.observation_dim - 2  # obs_dim minus xy dims
            current_full = np.concatenate([obs['achieved_goal'], obs['observation'][:body_dim]])
            current_norm = dataset.obs_normalizer.normalize(current_full).astype(np.float32)

            goal_full = _make_goal_obs(obs['desired_goal'], dataset.obs_normalizer)
            goal_norm = dataset.obs_normalizer.normalize(goal_full).astype(np.float32)

            cond = {
                0: torch.tensor(current_norm, device=device).unsqueeze(0),
                diffusion.horizon - 1: torch.tensor(goal_norm, device=device).unsqueeze(0),
            }
            with torch.no_grad():
                sample = diffusion.conditional_sample(cond, verbose=False)

            traj_norm = sample.trajectories[0].cpu().numpy()   # [H x transition_dim]
            acts_norm = traj_norm[:replan_every, :dataset.action_dim]
            action_buffer = list(dataset.act_normalizer.unnormalize(acts_norm))

        action = np.clip(action_buffer.pop(0), env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, info = env.step(action)

        if info.get('success', False):
            return True
        if terminated or truncated:
            return False

    return False


def evaluate(diffusion, dataset, env_id, *, n_episodes=100, replan_every=5,
             seeds=(0, 1, 2), device='cuda'):
    """
    Evaluate the diffusion model over multiple seeds.

    Uses the model in eval mode; restores train mode afterwards.

    Returns:
        normalized_score : mean success_rate * 100 across seeds
        seed_scores      : list of per-seed normalized scores
    """
    diffusion.eval()
    seed_scores = []
    all_eval_episodes = []  # for computing exploration metrics like coverage

    for seed in seeds:
        env = gym.make(env_id)
        successes = 0
        for ep in range(n_episodes):
            # seed only the first episode of each seed for reproducibility;
            # subsequent episodes draw random starts from the env's RNG
            ep_seed = seed if ep == 0 else None
            success, visited_xy = _rollout_episode(
                diffusion,
                dataset,
                env,
                replan_every,
                device,
                seed=ep_seed,
            )
            successes += int(success)
            eval_obs = np.zeros((len(visited_xy), dataset.observation_dim), dtype=np.float32)
            eval_obs[:, :2] = visited_xy
            all_eval_episodes.append({'observations': eval_obs})
        env.close()

        score = successes / n_episodes * 100
        seed_scores.append(score)
        print(f'  seed={seed}  successes={successes}/{n_episodes}  score={score:.1f}',
              flush=True)
    coverage_metrics = compute_xy_coverage_metrics_from_episodes(
        all_eval_episodes,
        x_bounds=(-4.0, 4.0),
        y_bounds=(-4.0, 4.0),
        bin_size=0.5,
        prefix='eval/xy',
    )

    diffusion.train()
    return {
        'normalized_score': float(np.mean(seed_scores)),
        'seed_scores': seed_scores,
        **coverage_metrics,
    }
