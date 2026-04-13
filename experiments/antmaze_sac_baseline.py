#!/usr/bin/env python3
"""
SAC baseline on AntMaze_UMaze-v5 (Gymnasium-Robotics) with W&B logging.

Metrics logged:
  - train/actor_loss, train/critic_loss, train/ent_coef, train/ent_coef_loss
  - rollout/ep_rew_mean, rollout/ep_len_mean
  - rollout/distance_to_goal  (mean distance between achieved and desired goal)
  - rollout/success_rate       (fraction of completed episodes that succeeded)
  - rollout/timesteps_to_goal  (mean steps taken in successful episodes)
  - time/fps, time/total_timesteps

Run:
    uv run python experiments/antmaze_sac_baseline.py
    uv run python experiments/antmaze_sac_baseline.py --total_timesteps 2000000 --seed 0
"""

import argparse
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers AntMaze envs

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Default hyperparameters ──────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "env_id": "AntMaze_UMaze-v5",
    "seed": 42,
    # SAC
    "total_timesteps": 1_000_000,
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    # HER
    "her_n_sampled_goal": 4,
    "her_goal_selection_strategy": "future",
    # Evaluation
    "eval_freq": 25_000,
    "n_eval_episodes": 20,
    # Logging
    "wandb_project": "gen-ai-world-models",
}


# ── Custom callback ──────────────────────────────────────────────────────────

class AntMazeMetricsCallback(BaseCallback):
    """
    Tracks AntMaze-specific metrics and forwards them to W&B.

    Logged at every `log_freq` environment steps:
      - rollout/distance_to_goal   mean L2 distance to goal across recent steps
      - rollout/success_rate        fraction of finished episodes that succeeded
      - rollout/timesteps_to_goal  mean steps taken by successful episodes
    """

    def __init__(self, log_freq: int = 2_000, verbose: int = 0):
        super().__init__(verbose)
        self._log_freq = log_freq
        # Per-step accumulator (cleared at each log flush)
        self._distances: list[float] = []
        # Per-episode accumulators (cleared at each log flush)
        self._ep_success: list[bool] = []
        self._ep_timesteps_to_goal: list[int] = []
        # Track how many steps each env has taken this episode
        self._ep_steps: list[int] = []

    def _on_training_start(self) -> None:
        self._ep_steps = [0] * self.training_env.num_envs

    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs", {})
        infos: list[dict] = self.locals.get("infos", [])
        dones: np.ndarray = self.locals.get("dones", np.zeros(self.training_env.num_envs, dtype=bool))

        for i in range(self.training_env.num_envs):
            self._ep_steps[i] += 1

            # Distance to goal from the current observation
            if isinstance(new_obs, dict) and "achieved_goal" in new_obs:
                dist = float(
                    np.linalg.norm(new_obs["achieved_goal"][i] - new_obs["desired_goal"][i])
                )
                self._distances.append(dist)

            # Episode completion
            if dones[i] and i < len(infos):
                success = bool(infos[i].get("success", False))
                self._ep_success.append(success)
                if success:
                    self._ep_timesteps_to_goal.append(self._ep_steps[i])
                self._ep_steps[i] = 0

        # Periodic flush to W&B
        if self.num_timesteps % self._log_freq == 0:
            self._flush()

        return True

    def _flush(self) -> None:
        log_dict: dict[str, float] = {}

        if self._distances:
            log_dict["rollout/distance_to_goal"] = float(np.mean(self._distances))

        if self._ep_success:
            log_dict["rollout/success_rate"] = float(np.mean(self._ep_success))

        if self._ep_timesteps_to_goal:
            log_dict["rollout/timesteps_to_goal"] = float(np.mean(self._ep_timesteps_to_goal))

        if log_dict:
            wandb.log(log_dict, step=self.num_timesteps)

        self._distances.clear()
        self._ep_success.clear()
        self._ep_timesteps_to_goal.clear()


# ── Environment factory ──────────────────────────────────────────────────────

def make_env(env_id: str, seed: int, rank: int = 0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    run = wandb.init(
        project=cfg["wandb_project"],
        name=f"sac-her-{cfg['env_id']}-seed{cfg['seed']}",
        config={k: v for k, v in cfg.items() if k != "wandb_project"},
        save_code=True,
    )

    # ── Environments ─────────────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(cfg["env_id"], cfg["seed"])])
    eval_env  = DummyVecEnv([make_env(cfg["env_id"], cfg["seed"] + 1000)])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": cfg["her_n_sampled_goal"],
            "goal_selection_strategy": cfg["her_goal_selection_strategy"],
        },
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        ent_coef=cfg["ent_coef"],
        verbose=1,
        seed=cfg["seed"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    antmaze_cb = AntMazeMetricsCallback(log_freq=2_000)

    wandb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_freq=cfg["eval_freq"],
        model_save_path=f"models/{run.id}",
        verbose=0,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{run.id}/best",
        log_path=f"logs/{run.id}",
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = CallbackList([antmaze_cb, wandb_cb, eval_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks,
        log_interval=4,
        progress_bar=True,
    )

    model.save(f"models/{run.id}/final_model")
    run.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC baseline on AntMaze_UMaze-v5")
    for key, val in DEFAULT_CONFIG.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", action="store_true", default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    main(vars(parser.parse_args()))
