"""
Shared exploration metrics for AntMaze-style experiments.

Assumptions:
- Episodes are dictionaries with:
    episode['observations'] : np.ndarray of shape [T, obs_dim]
- The first two observation dimensions are XY position.
"""

from typing import Dict, List, Tuple

import numpy as np


def compute_xy_coverage_metrics_from_episodes(
    episodes: List[dict],
    x_bounds: Tuple[float, float] = (-4.0, 4.0),
    y_bounds: Tuple[float, float] = (-4.0, 4.0),
    bin_size: float = 0.5,
    prefix: str = "xy",
) -> Dict[str, float]:
    """
    Compute simple XY exploration metrics from episode observations.

    Parameters
    ----------
    episodes:
        List of episode dicts, each with:
            'observations': [T, obs_dim]
        where obs[:, :2] = (x, y).
    x_bounds, y_bounds:
        Fixed bounding box for coverage calculation.
        Keep these fixed across runs for fair comparisons.
    bin_size:
        Width/height of each XY bin.
    prefix:
        Prefix for returned metric keys.

    Returns
    -------
    Dict[str, float]
        {
            f'{prefix}/unique_cells': ...,
            f'{prefix}/coverage_fraction': ...,
            f'{prefix}/visitation_entropy': ...,
        }
    """
    empty = {
        f"{prefix}/unique_cells": 0.0,
        f"{prefix}/coverage_fraction": 0.0,
        f"{prefix}/visitation_entropy": 0.0,
    }

    if not episodes:
        return empty

    all_xy = []
    for ep in episodes:
        obs = ep.get("observations")
        if obs is not None and len(obs) > 0:
            all_xy.append(obs[:, :2])

    if not all_xy:
        return empty

    xy = np.concatenate(all_xy, axis=0)

    xmin, xmax = x_bounds
    ymin, ymax = y_bounds

    # Keep only points within the fixed evaluation box.
    mask = (
        (xy[:, 0] >= xmin) & (xy[:, 0] < xmax) &
        (xy[:, 1] >= ymin) & (xy[:, 1] < ymax)
    )
    xy = xy[mask]

    if len(xy) == 0:
        return empty

    nx = int(np.ceil((xmax - xmin) / bin_size))
    ny = int(np.ceil((ymax - ymin) / bin_size))
    total_cells = nx * ny

    ix = np.floor((xy[:, 0] - xmin) / bin_size).astype(int)
    iy = np.floor((xy[:, 1] - ymin) / bin_size).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    flat_idx = ix * ny + iy
    unique_cells, counts = np.unique(flat_idx, return_counts=True)

    unique_count = float(len(unique_cells))
    coverage_fraction = float(unique_count / total_cells)

    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    return {
        f"{prefix}/unique_cells": unique_count,
        f"{prefix}/coverage_fraction": coverage_fraction,
        f"{prefix}/visitation_entropy": entropy,
    }