"""Particle advection through a velocity field with Runge-Kutta integration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from flowgp.field import VelocityField


def make_interpolator(field: VelocityField) -> Callable[[np.ndarray], np.ndarray]:
    """Return a bilinear interpolator mapping points ``(m, 2)`` to velocities.

    Points outside the grid are clamped to the boundary so particles that drift
    off the domain still receive a defined velocity.
    """
    interp_u = RegularGridInterpolator(
        (field.ys, field.xs), field.u, bounds_error=False, fill_value=None
    )
    interp_v = RegularGridInterpolator(
        (field.ys, field.xs), field.v, bounds_error=False, fill_value=None
    )
    x_lo, x_hi = field.xs[0], field.xs[-1]
    y_lo, y_hi = field.ys[0], field.ys[-1]

    def velocity(points: np.ndarray) -> np.ndarray:
        clamped = np.column_stack(
            [
                np.clip(points[:, 1], y_lo, y_hi),  # RegularGridInterpolator wants (y, x)
                np.clip(points[:, 0], x_lo, x_hi),
            ]
        )
        u = interp_u(clamped)
        v = interp_v(clamped)
        return np.stack([u, v], axis=1)

    return velocity


def rk4_step(
    points: np.ndarray, velocity: Callable[[np.ndarray], np.ndarray], dt: float
) -> np.ndarray:
    """Advance ``points`` one step of size ``dt`` with fourth-order Runge-Kutta."""
    k1 = velocity(points)
    k2 = velocity(points + 0.5 * dt * k1)
    k3 = velocity(points + 0.5 * dt * k2)
    k4 = velocity(points + dt * k3)
    return points + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def advect_particles(
    field: VelocityField,
    start_points: np.ndarray,
    n_steps: int = 100,
    dt: float = 0.05,
    velocity: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Advect particles through the field and return their trajectories.

    Args:
        field: The velocity field to move through.
        start_points: Initial positions, shape ``(p, 2)``.
        n_steps: Number of integration steps.
        dt: Step size.
        velocity: Optional velocity function (for example a GP reconstruction);
            defaults to bilinear interpolation of ``field``.

    Returns:
        Trajectories of shape ``(n_steps + 1, p, 2)``.
    """
    if velocity is None:
        velocity = make_interpolator(field)
    positions = np.asarray(start_points, dtype=float).copy()
    trajectory = np.empty((n_steps + 1, positions.shape[0], 2))
    trajectory[0] = positions
    for step in range(n_steps):
        positions = rk4_step(positions, velocity, dt)
        trajectory[step + 1] = positions
    return trajectory
