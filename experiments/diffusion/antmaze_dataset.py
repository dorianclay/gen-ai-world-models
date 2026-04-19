"""
Antmaze offline dataset loader.

Loads D4RL antmaze HDF5 files directly — no gym, mujoco-py, or d4rl required.
Implements GoalDataset semantics: conditions on both the first and last observation
of each trajectory window (start position and goal).
"""
import os
import collections
from collections import namedtuple

import gymnasium as gym
import gymnasium_robotics  # noqa: registers AntMaze envs

import h5py
import numpy as np
import torch
import requests
from tqdm import tqdm


Batch = namedtuple('Batch', 'trajectories conditions')

# HDF5 URLs for D4RL antmaze datasets
DATASET_URLS = {
    'antmaze-umaze-v2': (
        'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/'
        'Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5'
    ),
    'antmaze-umaze-diverse-v2': (
        'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/'
        'Ant_maze_u-maze_noisy_multistart_True_multigoal_True_sparse.hdf5'
    ),
    'antmaze-medium-play-v2': (
        'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/'
        'Ant_maze_medium_noisy_multistart_False_multigoal_False_sparse.hdf5'
    ),
    'antmaze-large-play-v2': (
        'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/'
        'Ant_maze_large_noisy_multistart_False_multigoal_False_sparse.hdf5'
    ),
    'antmaze-big-diverse-v2': (
        'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/'
        'Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse.hdf5'
    ),
    'antmaze-hardest-diverse-v2': (
        'https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/'
        'Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse.hdf5'
    )
}


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class GaussianNormalizer:
    """Normalizes data to zero mean, unit variance per dimension."""

    def __init__(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        # avoid dividing by zero for constant dimensions
        self.std[self.std < 1e-4] = 1.0

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean


# ---------------------------------------------------------------------------
# HDF5 loading and episode splitting
# ---------------------------------------------------------------------------

def download_dataset(env_name, cache_dir=None):
    """
    Download the antmaze HDF5 file if not already cached.
    Returns the local path to the file.
    """
    if env_name not in DATASET_URLS:
        raise ValueError(
            f'Unknown dataset "{env_name}". Available: {list(DATASET_URLS)}'
        )
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.datasets/antmaze')
    os.makedirs(cache_dir, exist_ok=True)

    filename = DATASET_URLS[env_name].split('/')[-1]
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        print(f'[ antmaze_dataset ] Using cached dataset at {local_path}')
        return local_path

    url = DATASET_URLS[env_name]
    print(f'[ antmaze_dataset ] Downloading {env_name} from {url}')
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(local_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    return local_path


def load_h5_dataset(path):
    """Load flat arrays from an HDF5 file."""
    with h5py.File(path, 'r') as f:
        data = {
            key: f[key][:]
            for key in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']
            if key in f
        }
    return data


def split_into_episodes(flat_data, max_path_length=700):
    """
    Split flat (N,) arrays into a list of episode dicts.

    An episode ends when terminals[i] or timeouts[i] is True.
    Episodes longer than max_path_length are truncated.
    """
    N = flat_data['rewards'].shape[0]
    use_timeouts = 'timeouts' in flat_data

    episodes = []
    buf = collections.defaultdict(list)
    step = 0

    for i in range(N):
        for key, arr in flat_data.items():
            buf[key].append(arr[i])
        step += 1

        done = bool(flat_data['terminals'][i])
        timeout = bool(flat_data['timeouts'][i]) if use_timeouts else (step >= max_path_length)

        if done or timeout or step >= max_path_length:
            episode = {k: np.stack(v) for k, v in buf.items()}
            episodes.append(episode)
            buf = collections.defaultdict(list)
            step = 0

    # flush any remaining partial episode
    if buf['rewards']:
        episode = {k: np.stack(v) for k, v in buf.items()}
        episodes.append(episode)

    return episodes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AntmazeDataset(torch.utils.data.Dataset):
    """
    GoalDataset for antmaze: each item is a trajectory window of length `horizon`
    with conditions on both the first observation (start) and the last (goal).

    Args:
        path_or_name : local HDF5 path OR a dataset name (e.g. 'antmaze-umaze-v2')
                       that will be auto-downloaded.
        horizon      : trajectory window length (default 128, as in the diffuser paper).
        max_path_length : episode timeout length (default 700 for antmaze-umaze).
        use_padding  : if True, pad short episodes so windows can start near the end.
        cache_dir    : directory for downloaded datasets.
    """

    def __init__(
        self,
        path_or_name,
        horizon=128,
        max_path_length=700,
        use_padding=True,
        cache_dir=None,
    ):
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # resolve path
        if os.path.isfile(path_or_name):
            h5_path = path_or_name
        else:
            h5_path = download_dataset(path_or_name, cache_dir=cache_dir)

        print(f'[ AntmazeDataset ] Loading {h5_path}')
        flat_data = load_h5_dataset(h5_path)
        episodes = split_into_episodes(flat_data, max_path_length=max_path_length)
        print(f'[ AntmazeDataset ] {len(episodes)} episodes loaded')

        # stack episodes into padded arrays [n_episodes x max_path_length x dim]
        self.observation_dim = flat_data['observations'].shape[-1]
        self.action_dim = flat_data['actions'].shape[-1]
        n_ep = len(episodes)

        obs_arr = np.zeros((n_ep, max_path_length, self.observation_dim), dtype=np.float32)
        act_arr = np.zeros((n_ep, max_path_length, self.action_dim), dtype=np.float32)
        path_lengths = np.zeros(n_ep, dtype=int)

        for i, ep in enumerate(episodes):
            L = min(len(ep['observations']), max_path_length)
            obs_arr[i, :L] = ep['observations'][:L]
            act_arr[i, :L] = ep['actions'][:L]
            path_lengths[i] = L

        # compute normalizers over all actual data
        all_obs = np.concatenate([obs_arr[i, :path_lengths[i]] for i in range(n_ep)], axis=0)
        all_act = np.concatenate([act_arr[i, :path_lengths[i]] for i in range(n_ep)], axis=0)
        self.obs_normalizer = GaussianNormalizer(all_obs)
        self.act_normalizer = GaussianNormalizer(all_act)

        # normalize in place
        for i in range(n_ep):
            L = path_lengths[i]
            obs_arr[i, :L] = self.obs_normalizer.normalize(obs_arr[i, :L])
            act_arr[i, :L] = self.act_normalizer.normalize(act_arr[i, :L])

        self.observations = obs_arr  # [n_ep x max_path_length x obs_dim]
        self.actions = act_arr       # [n_ep x max_path_length x act_dim]
        self.path_lengths = path_lengths

        self.indices = self._make_indices(path_lengths, horizon)
        print(f'[ AntmazeDataset ] {len(self.indices)} trajectory windows')

    def _make_indices(self, path_lengths, horizon):
        """
        Build a list of (episode_idx, start, end) tuples covering all
        valid trajectory windows.
        """
        indices = []
        for ep_i, length in enumerate(path_lengths):
            # latest valid start so that [start, start+horizon) fits within the episode
            max_start = min(length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, length - horizon)
            for start in range(max_start):
                indices.append((ep_i, start, start + horizon))
        return np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_i, start, end = self.indices[idx]

        obs = self.observations[ep_i, start:end]   # [horizon x obs_dim]
        act = self.actions[ep_i, start:end]         # [horizon x act_dim]

        # trajectory: [horizon x (action_dim + obs_dim)]  — actions first
        trajectory = np.concatenate([act, obs], axis=-1).astype(np.float32)

        # GoalDataset: condition on the first and last observation in the window
        conditions = {
            0: obs[0].astype(np.float32),
            self.horizon - 1: obs[-1].astype(np.float32),
        }

        return Batch(trajectory, conditions)


# ---------------------------------------------------------------------------
# Online dataset (gymnasium rollouts)
# ---------------------------------------------------------------------------

# Observation dimension in D4RL offline datasets (2 xy + 27 body state = 29).
# Pass this to OnlineAntmazeDataset so online observations match offline shape,
# keeping model checkpoints compatible across both training modes.
DATASET_OBS_DIM = {
    'antmaze-umaze-v2':           29,
    'antmaze-umaze-diverse-v2':   29,
    'antmaze-medium-play-v2':     29,
    'antmaze-large-play-v2':      29,
    'antmaze-big-diverse-v2':     29,
    'antmaze-hardest-diverse-v2': 29,
}


def _make_goal_obs(desired_goal, obs_normalizer):
    """Build unnormalized goal obs: target xy + dataset-mean body state (→ 0 after norm)."""
    goal = obs_normalizer.mean.copy()
    goal[:2] = desired_goal
    return goal


def collect_episodes(env_id, n_episodes, *, diffusion=None, dataset=None,
                     observation_dim=None, max_path_length=700, replan_every=5,
                     seed=None, device='cpu'):
    """
    Collect episodes from a gymnasium AntMaze environment.

    Random policy when diffusion is None; MPC planning otherwise (requires
    dataset for its frozen normalizers).  observation_dim slices obs['observation']
    to match a specific observation space (None = use full gym observation).

    Returns a list of raw (unnormalized) episode dicts with keys
    'observations' [T x obs_dim] and 'actions' [T x act_dim].
    """
    if diffusion is not None:
        assert dataset is not None, 'dataset is required for normalizers when using diffusion'
        diffusion.eval()

    body_dim = (observation_dim - 2) if observation_dim is not None else None
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    episodes = []

    for i in tqdm(range(n_episodes), desc='Collecting episodes'):
        ep_seed = int(rng.integers(0, 2**31)) if seed is not None else None
        obs_dict, _ = env.reset(seed=ep_seed if i == 0 else None)

        ep_obs, ep_acts = [], []
        action_buffer = []

        for _ in range(max_path_length):
            body = obs_dict['observation'] if body_dim is None else obs_dict['observation'][:body_dim]
            full_obs = np.concatenate([obs_dict['achieved_goal'], body]).astype(np.float32)

            if diffusion is not None:
                if not action_buffer:
                    current_norm = dataset.obs_normalizer.normalize(full_obs).astype(np.float32)
                    goal_full = _make_goal_obs(obs_dict['desired_goal'], dataset.obs_normalizer)
                    goal_norm = dataset.obs_normalizer.normalize(goal_full).astype(np.float32)
                    cond = {
                        0: torch.tensor(current_norm, device=device).unsqueeze(0),
                        diffusion.horizon - 1: torch.tensor(goal_norm, device=device).unsqueeze(0),
                    }
                    with torch.no_grad():
                        sample = diffusion.conditional_sample(cond, verbose=False)
                    traj_norm = sample.trajectories[0].cpu().numpy()
                    acts_norm = traj_norm[:replan_every, :dataset.action_dim]
                    action_buffer = list(dataset.act_normalizer.unnormalize(acts_norm))
                action = np.clip(action_buffer.pop(0), env.action_space.low, env.action_space.high).astype(np.float32)
            else:
                action = env.action_space.sample().astype(np.float32)

            ep_obs.append(full_obs)
            ep_acts.append(action)
            obs_dict, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        if ep_obs:
            episodes.append({
                'observations': np.stack(ep_obs),
                'actions': np.stack(ep_acts),
            })

    env.close()
    if diffusion is not None:
        diffusion.train()
    return episodes


class OnlineAntmazeDataset(torch.utils.data.Dataset):
    """
    Dataset built from live gymnasium AntMaze rollouts.

    Warmup: collects n_warmup_episodes with a random policy to seed the dataset
    and fit normalizers. Normalizers are frozen after warmup so that refreshes
    remain compatible with the checkpoint being fine-tuned.

    refresh(episodes): replaces the current data with new pre-collected episodes
    (normalized with the frozen normalizers) and rebuilds trajectory windows.
    The Trainer's dataloader must be reset afterwards via trainer.reset_dataloader().
    """

    def __init__(
        self,
        env_id,
        horizon=128,
        n_warmup_episodes=1000,
        max_path_length=700,
        use_padding=True,
        observation_dim=None,
        seed=None,
    ):
        self.env_id = env_id
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        print(f'[ OnlineAntmazeDataset ] Warmup: {n_warmup_episodes} random episodes from {env_id}')
        warmup = collect_episodes(
            env_id, n_warmup_episodes,
            observation_dim=observation_dim,
            max_path_length=max_path_length,
            seed=seed,
        )
        print(f'[ OnlineAntmazeDataset ] Collected {len(warmup)} warmup episodes')

        self.observation_dim = warmup[0]['observations'].shape[-1]
        self.action_dim = warmup[0]['actions'].shape[-1]

        # Fit normalizers once from warmup data and freeze for the lifetime of this dataset.
        all_obs = np.concatenate([ep['observations'] for ep in warmup], axis=0)
        all_act = np.concatenate([ep['actions'] for ep in warmup], axis=0)
        self.obs_normalizer = GaussianNormalizer(all_obs)
        self.act_normalizer = GaussianNormalizer(all_act)

        self._build_from_episodes(warmup)

    def _build_from_episodes(self, episodes):
        """Normalize episodes with frozen normalizers and rebuild trajectory windows."""
        n_ep = len(episodes)
        obs_arr = np.zeros((n_ep, self.max_path_length, self.observation_dim), dtype=np.float32)
        act_arr = np.zeros((n_ep, self.max_path_length, self.action_dim), dtype=np.float32)
        path_lengths = np.zeros(n_ep, dtype=int)

        for i, ep in enumerate(episodes):
            L = min(len(ep['observations']), self.max_path_length)
            obs_arr[i, :L] = self.obs_normalizer.normalize(ep['observations'][:L])
            act_arr[i, :L] = self.act_normalizer.normalize(ep['actions'][:L])
            path_lengths[i] = L

        self.observations = obs_arr
        self.actions = act_arr
        self.path_lengths = path_lengths
        self.indices = self._make_indices(path_lengths, self.horizon)
        print(f'[ OnlineAntmazeDataset ] {len(self.indices)} windows from {n_ep} episodes')

    def refresh(self, episodes):
        """Replace dataset contents with new episodes; normalizers stay frozen."""
        self._build_from_episodes(episodes)

    def _make_indices(self, path_lengths, horizon):
        indices = []
        for ep_i, length in enumerate(path_lengths):
            max_start = min(length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, length - horizon)
            for start in range(max_start):
                indices.append((ep_i, start, start + horizon))
        return np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_i, start, end = self.indices[idx]
        obs = self.observations[ep_i, start:end]
        act = self.actions[ep_i, start:end]
        trajectory = np.concatenate([act, obs], axis=-1).astype(np.float32)
        conditions = {
            0: obs[0].astype(np.float32),
            self.horizon - 1: obs[-1].astype(np.float32),
        }
        return Batch(trajectory, conditions)
