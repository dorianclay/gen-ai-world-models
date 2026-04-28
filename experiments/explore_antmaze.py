"""
Quick exploration of the antmaze-umaze-v2 dataset.
Shows raw HDF5 structure, episode splits, and a few dataset batches.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from experiments.diffusion.antmaze_dataset import (
    download_dataset, load_h5_dataset, split_into_episodes, AntmazeDataset
)

ENV = 'antmaze-umaze-v2'

# Raw HDF5
print('\n=== Raw HDF5 ===')
h5_path = download_dataset(ENV)
flat = load_h5_dataset(h5_path)
for key, arr in flat.items():
    print(f'  {key:14s}  shape={arr.shape}  dtype={arr.dtype}')

print('\nFirst 3 raw observations (xy + body state):')
for i, obs in enumerate(flat['observations'][:3]):
    print(f'  [{i}] xy=({obs[0]:.2f}, {obs[1]:.2f})  body[0:4]={obs[2:6].round(3)}')

print('\nFirst 3 raw actions:')
for i, act in enumerate(flat['actions'][:3]):
    print(f'  [{i}] {act.round(3)}')

# Episode spltis
print('\n=== Episodes ===')
episodes = split_into_episodes(flat)
lengths = [len(ep['rewards']) for ep in episodes]
print(f'  Total episodes : {len(episodes)}')
print(f'  Length min/max : {min(lengths)} / {max(lengths)}')
print(f'  Length mean    : {np.mean(lengths):.1f}')

for i in range(3):
    ep = episodes[i]
    xy_start = ep['observations'][0, :2].round(2)
    xy_end   = ep['observations'][-1, :2].round(2)
    n_reward = ep['rewards'].sum()
    print(f'  ep[{i}]  len={len(ep["rewards"]):3d}  start_xy={xy_start}  end_xy={xy_end}  total_reward={n_reward:.0f}')

# Batches
print('\n=== AntmazeDataset batches ===')
ds = AntmazeDataset(ENV, horizon=64)
print(f'  Dataset size (windows) : {len(ds)}')
print(f'  observation_dim        : {ds.observation_dim}')
print(f'  action_dim             : {ds.action_dim}')
print(f'  horizon                : {ds.horizon}')

for i in range(3):
    batch = ds[i]
    traj = batch.trajectories   # [horizon x (act_dim + obs_dim)]
    cond = batch.conditions     # {0: start_obs, horizon-1: goal_obs}
    act  = traj[:, :ds.action_dim]
    obs  = traj[:, ds.action_dim:]
    print(f'\n  batch[{i}]')
    print(f'    trajectory shape : {traj.shape}')
    print(f'    start_obs (norm) : xy=({cond[0][0]:.3f}, {cond[0][1]:.3f})')
    print(f'    goal_obs  (norm) : xy=({cond[ds.horizon-1][0]:.3f}, {cond[ds.horizon-1][1]:.3f})')
    print(f'    obs  mean/std    : {obs.mean():.3f} / {obs.std():.3f}')
    print(f'    act  mean/std    : {act.mean():.3f} / {act.std():.3f}')
