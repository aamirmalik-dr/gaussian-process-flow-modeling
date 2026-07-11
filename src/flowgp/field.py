"""A synthetic, divergence-free 2D velocity field.

The field is defined through a stream function ``psi(x, y)``. Taking the velocity
as ``(u, v) = (d psi / d y, -d psi / d x)`` guarantees that the flow is
divergence free (incompressible) analytically, which gives the particle-advection
demo a physically meaningful field and gives the tests a property to check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityField:
    """A velocity field sampled on a regular grid.

    Attributes:
        xs: Grid x coordinates, shape ``(nx,)``.
        ys: Grid y coordinates, shape ``(ny,)``.
        u: x-velocity on the grid, shape ``(ny, nx)``.
        v: y-velocity on the grid, shape ``(ny, nx)``.
    """

    xs: np.ndarray
    ys: np.ndarray
    u: np.ndarray
    v: np.ndarray

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return float(self.xs[0]), float(self.xs[-1]), float(self.ys[0]), float(self.ys[-1])


def _velocity_at(x: np.ndarray, y: np.ndarray, modes: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Analytic velocity from the stream function at arbitrary points.

    The stream function is a sum of sinusoidal modes,
    ``psi = sum_k a_k sin(fx_k x + px_k) sin(fy_k y + py_k)``. The velocity is its
    analytic curl, so no numerical differentiation is needed.
    """
    u = np.zeros_like(x, dtype=float)
    v = np.zeros_like(x, dtype=float)
    for a, fx, fy, px, py in modes:
        sx = np.sin(fx * x + px)
        cx = np.cos(fx * x + px)
        sy = np.sin(fy * y + py)
        cy = np.cos(fy * y + py)
        # u = d psi / d y ; v = -d psi / d x
        u += a * sx * fy * cy
        v += -a * fx * cx * sy
    return u, v


def synthetic_flow_field(
    n: int = 40, extent: float = 2 * np.pi, n_modes: int = 4, seed: int = 0
) -> VelocityField:
    """Build a divergence-free velocity field on an ``n`` by ``n`` grid.

    Args:
        n: Grid resolution per axis.
        extent: The domain is ``[0, extent] x [0, extent]``.
        n_modes: Number of sinusoidal stream-function modes.
        seed: Random seed for the mode parameters.

    Returns:
        The sampled :class:`VelocityField`.
    """
    rng = np.random.default_rng(seed)
    modes = []
    for _ in range(n_modes):
        a = rng.uniform(0.5, 1.5)
        fx = rng.integers(1, 3)
        fy = rng.integers(1, 3)
        px = rng.uniform(0, 2 * np.pi)
        py = rng.uniform(0, 2 * np.pi)
        modes.append((a, fx, fy, px, py))

    xs = np.linspace(0, extent, n)
    ys = np.linspace(0, extent, n)
    gx, gy = np.meshgrid(xs, ys)
    u, v = _velocity_at(gx, gy, modes)
    field = VelocityField(xs=xs, ys=ys, u=u, v=v)
    # Stash the modes so callers can sample the analytic field off-grid.
    field.modes = modes  # type: ignore[attr-defined]
    return field


def sample_velocity(field: VelocityField, points: np.ndarray) -> np.ndarray:
    """Sample the analytic velocity at arbitrary ``points`` of shape ``(m, 2)``.

    Raises:
        AttributeError: If the field was not produced by
            :func:`synthetic_flow_field` (analytic modes unavailable).
    """
    modes = getattr(field, "modes", None)
    if modes is None:
        raise AttributeError("sample_velocity needs a field from synthetic_flow_field")
    u, v = _velocity_at(points[:, 0], points[:, 1], modes)
    return np.stack([u, v], axis=1)


def divergence(field: VelocityField) -> np.ndarray:
    """Numerical divergence ``du/dx + dv/dy`` of the sampled field.

    For the analytic divergence-free construction this is near zero away from the
    grid boundaries, up to finite-difference error.
    """
    dx = field.xs[1] - field.xs[0]
    dy = field.ys[1] - field.ys[0]
    du_dx = np.gradient(field.u, dx, axis=1)
    dv_dy = np.gradient(field.v, dy, axis=0)
    return du_dx + dv_dy
