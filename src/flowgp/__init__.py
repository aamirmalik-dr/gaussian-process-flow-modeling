"""Gaussian process regression and particle flow over a 2D velocity field.

The package builds a reproducible, divergence-free (incompressible) 2D velocity
field from a stream function, reconstructs that field from sparse noisy samples
with Gaussian process regression, and advects particles through the field with a
fourth-order Runge-Kutta integrator. This is an ocean-current-style
spatiotemporal modeling demo that needs no external dataset.
"""

from flowgp.advection import advect_particles, rk4_step
from flowgp.data import make_observations, train_test_split_points
from flowgp.field import (
    VelocityField,
    divergence,
    synthetic_flow_field,
)
from flowgp.gp import GPFieldModel

__all__ = [
    "VelocityField",
    "synthetic_flow_field",
    "divergence",
    "make_observations",
    "train_test_split_points",
    "GPFieldModel",
    "advect_particles",
    "rk4_step",
]

__version__ = "0.1.0"
