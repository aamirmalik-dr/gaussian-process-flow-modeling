"""Reconstruct a velocity field with a GP and advect particles through it.

Builds a synthetic divergence-free field, samples sparse noisy observations,
fits a Gaussian process to reconstruct the field, reports reconstruction error
and the field's divergence, then advects particles through both the true and
reconstructed fields and writes figures.

Usage:
    python scripts/simulate.py --n-obs 150 --grid 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flowgp.advection import advect_particles
from flowgp.data import make_observations, train_test_split_points
from flowgp.field import divergence, synthetic_flow_field
from flowgp.gp import GPFieldModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--n-obs", type=int, default=150)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    field = synthetic_flow_field(n=args.grid, seed=args.seed)
    div = divergence(field)
    interior = div[2:-2, 2:-2]
    dx = field.xs[1] - field.xs[0]
    derivative_scale = np.abs(np.gradient(field.u, dx, axis=1)).mean()
    rel_div = np.abs(interior).mean() / derivative_scale
    print(f"Field divergence: mean |div| interior = {np.abs(interior).mean():.3e}, "
          f"which is {rel_div:.1%} of the mean derivative magnitude "
          f"(the field is analytically divergence free; this residual is finite-difference error)")

    points, velocities = make_observations(field, n_obs=args.n_obs, noise=args.noise, seed=args.seed)
    p_tr, v_tr, p_te, v_te = train_test_split_points(points, velocities, seed=args.seed)

    gp = GPFieldModel(length_scale=1.0, seed=args.seed).fit(p_tr, v_tr)
    test_rmse = gp.score_rmse(p_te, v_te)
    print(f"GP reconstruction RMSE on held-out points: {test_rmse:.4f} "
          f"(from {len(p_tr)} noisy observations)")

    # Grid reconstruction error against the true grid field.
    gx, gy = np.meshgrid(field.xs, field.ys)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    mean, std = gp.predict(grid_points, return_std=True)
    true_grid = np.column_stack([field.u.ravel(), field.v.ravel()])
    grid_rmse = float(np.sqrt(np.mean(np.sum((mean - true_grid) ** 2, axis=1))))
    print(f"GP reconstruction RMSE over full grid: {grid_rmse:.4f}")

    # Advect a line of particles through the true field.
    starts = np.column_stack(
        [np.full(12, field.xs[len(field.xs) // 6]), np.linspace(field.ys[2], field.ys[-3], 12)]
    )
    traj_true = advect_particles(field, starts, n_steps=200, dt=0.03)

    # Figures.
    plt.figure(figsize=(6, 5))
    plt.quiver(gx, gy, field.u, field.v, np.hypot(field.u, field.v), cmap="viridis")
    for p in range(traj_true.shape[1]):
        plt.plot(traj_true[:, p, 0], traj_true[:, p, 1], color="crimson", linewidth=1)
    plt.title("Velocity field and particle trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_dir / "field_and_trajectories.png", dpi=120)
    plt.close()

    std_mag = np.hypot(std[:, 0], std[:, 1]).reshape(field.u.shape)
    plt.figure(figsize=(6, 5))
    plt.imshow(std_mag, origin="lower", extent=field.extent, cmap="magma")
    plt.colorbar(label="GP predictive std")
    plt.scatter(p_tr[:, 0], p_tr[:, 1], s=10, c="cyan", label="observations")
    plt.title("Reconstruction uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "uncertainty.png", dpi=120)
    plt.close()

    print(f"Wrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
