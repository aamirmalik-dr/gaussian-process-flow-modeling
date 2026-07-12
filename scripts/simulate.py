"""Reconstruct a velocity field with a GP and advect particles through it.

Builds a synthetic divergence-free field, samples sparse noisy observations,
fits a Gaussian process to reconstruct the field, reports reconstruction error
and calibration, then advects particles through the field and writes the hero
uncertainty map, a trajectory figure, a metrics file, and the fitted model.

Usage:
    python scripts/simulate.py --n-obs 200 --grid 45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flowgp.advection import advect_particles
from flowgp.data import make_observations, train_test_split_points
from flowgp.field import divergence, sample_velocity, synthetic_flow_field
from flowgp.gp import GPFieldModel
from flowgp.metrics import coverage, nlpd, rmse
from flowgp.persistence import save_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=45)
    parser.add_argument("--n-obs", type=int, default=200)
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
    mean_abs_div = float(np.abs(interior).mean())
    print(
        f"Field divergence: mean |div| interior = {mean_abs_div:.3e}, "
        f"which is {rel_div:.1%} of the mean derivative magnitude "
        f"(the field is analytically divergence free; this residual is finite-difference error)"
    )

    points, velocities = make_observations(
        field, n_obs=args.n_obs, noise=args.noise, seed=args.seed
    )
    p_tr, v_tr, p_te, v_te = train_test_split_points(points, velocities, seed=args.seed)

    gp = GPFieldModel(length_scale=1.0, seed=args.seed).fit(p_tr, v_tr)

    # Held-out accuracy and calibration against noise-free ground truth.
    true_te = sample_velocity(field, p_te)
    mean_te, std_te = gp.predict(p_te, return_std=True)
    test_rmse = rmse(mean_te, true_te)
    test_nlpd = nlpd(mean_te, std_te, true_te)
    cov1 = coverage(mean_te, std_te, true_te, k=1.0)
    cov2 = coverage(mean_te, std_te, true_te, k=2.0)
    lml = gp.log_marginal_likelihood()
    print(
        f"GP held-out RMSE vs ground truth: {test_rmse:.4f} (from {len(p_tr)} noisy observations)"
    )
    print(f"GP held-out NLPD: {test_nlpd:.4f}; coverage 1-sigma {cov1:.3f}, 2-sigma {cov2:.3f}")
    print(f"GP log marginal likelihood (u+v): {lml:.2f}")

    # Full-grid reconstruction and posterior standard deviation.
    gx, gy = np.meshgrid(field.xs, field.ys)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    mean, std = gp.predict(grid_points, return_std=True)
    true_grid = np.column_stack([field.u.ravel(), field.v.ravel()])
    grid_rmse = rmse(mean, true_grid)
    print(f"GP reconstruction RMSE over full grid: {grid_rmse:.4f}")

    u_rec = mean[:, 0].reshape(field.u.shape)
    v_rec = mean[:, 1].reshape(field.v.shape)
    std_mag = np.hypot(std[:, 0], std[:, 1]).reshape(field.u.shape)

    # Advect a line of particles through the true field.
    starts = np.column_stack(
        [np.full(12, field.xs[len(field.xs) // 6]), np.linspace(field.ys[2], field.ys[-3], 12)]
    )
    traj_true = advect_particles(field, starts, n_steps=200, dt=0.03)

    # Hero figure: GP posterior uncertainty map with reconstructed streamlines
    # and the observation sites overlaid. Uncertainty is low near observations
    # and grows into the gaps, exactly where a reconstruction should be unsure.
    x0, x1, y0, y1 = field.extent
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(std_mag, origin="lower", extent=field.extent, cmap="magma", aspect="auto")
    ax.streamplot(gx, gy, u_rec, v_rec, color="white", density=1.1, linewidth=0.7, arrowsize=0.8)
    ax.scatter(
        p_tr[:, 0],
        p_tr[:, 1],
        s=14,
        c="cyan",
        edgecolors="black",
        linewidths=0.3,
        label="observations",
        zorder=3,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("GP predictive standard deviation")
    ax.set_title("Reconstruction uncertainty over the reconstructed flow")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_map.png", dpi=130)
    plt.close(fig)

    # Secondary: true field with particle trajectories.
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

    # Persist metrics and the fitted model.
    metrics = {
        "grid": args.grid,
        "n_obs": args.n_obs,
        "n_train": int(len(p_tr)),
        "n_test": int(len(p_te)),
        "noise": args.noise,
        "seed": args.seed,
        "mean_abs_divergence_interior": round(mean_abs_div, 6),
        "divergence_fraction_of_gradient": round(float(rel_div), 5),
        "test_rmse": round(test_rmse, 6),
        "grid_rmse": round(grid_rmse, 6),
        "test_nlpd": round(test_nlpd, 6),
        "coverage_1sigma": round(cov1, 4),
        "coverage_2sigma": round(cov2, 4),
        "log_marginal_likelihood": round(lml, 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_model(gp, out_dir / "gp_flow_model.joblib")

    print(f"Wrote figures, metrics.json, and gp_flow_model.joblib to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
