"""Sampling observations from a velocity field and splitting them."""

from __future__ import annotations

import numpy as np

from flowgp.field import VelocityField, sample_velocity


def make_observations(
    field: VelocityField, n_obs: int = 120, noise: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Draw noisy velocity observations at random points in the domain.

    Args:
        field: A field produced by ``synthetic_flow_field`` (analytic sampling).
        n_obs: Number of observation points.
        noise: Standard deviation of additive Gaussian noise on the velocities.
        seed: Random seed.

    Returns:
        A tuple ``(points, velocities)`` of shapes ``(n_obs, 2)`` each.
    """
    rng = np.random.default_rng(seed)
    x_lo, x_hi = field.xs[0], field.xs[-1]
    y_lo, y_hi = field.ys[0], field.ys[-1]
    points = np.column_stack([rng.uniform(x_lo, x_hi, n_obs), rng.uniform(y_lo, y_hi, n_obs)])
    velocities = sample_velocity(field, points)
    velocities = velocities + rng.normal(0, noise, velocities.shape)
    return points, velocities


def train_test_split_points(
    points: np.ndarray, velocities: np.ndarray, test_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split observation points into train and test subsets."""
    rng = np.random.default_rng(seed)
    n = len(points)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return points[train_idx], velocities[train_idx], points[test_idx], velocities[test_idx]
