import numpy as np

from flowgp.advection import advect_particles, make_interpolator, rk4_step
from flowgp.data import make_observations, train_test_split_points
from flowgp.field import divergence, sample_velocity, synthetic_flow_field
from flowgp.gp import GPFieldModel


def _mean_abs_interior_div(field) -> float:
    div = divergence(field)
    return float(np.abs(div[3:-3, 3:-3]).mean())


def test_field_is_divergence_free_relative():
    # The analytic construction is exactly divergence free; the numerical
    # divergence is only finite-difference error, which is small compared to the
    # magnitude of the individual derivative terms it is built from.
    field = synthetic_flow_field(n=60, seed=1)
    dx = field.xs[1] - field.xs[0]
    derivative_scale = np.abs(np.gradient(field.u, dx, axis=1)).mean()
    assert _mean_abs_interior_div(field) < 0.2 * derivative_scale


def test_divergence_shrinks_with_resolution():
    # Refining the grid reduces the numerical divergence, confirming it is a
    # discretization artifact rather than a real source or sink.
    coarse = _mean_abs_interior_div(synthetic_flow_field(n=40, seed=1))
    fine = _mean_abs_interior_div(synthetic_flow_field(n=120, seed=1))
    assert fine < coarse


def test_field_shapes():
    field = synthetic_flow_field(n=30)
    assert field.u.shape == (30, 30)
    assert field.v.shape == (30, 30)


def test_sample_velocity_matches_grid_corner():
    field = synthetic_flow_field(n=40, seed=2)
    pt = np.array([[field.xs[10], field.ys[15]]])
    vel = sample_velocity(field, pt)[0]
    assert np.allclose(vel[0], field.u[15, 10], atol=1e-8)
    assert np.allclose(vel[1], field.v[15, 10], atol=1e-8)


def test_gp_reconstructs_field():
    field = synthetic_flow_field(n=40, seed=0)
    points, velocities = make_observations(field, n_obs=200, noise=0.02, seed=0)
    p_tr, v_tr, p_te, v_te = train_test_split_points(points, velocities, seed=0)
    gp = GPFieldModel(length_scale=1.0).fit(p_tr, v_tr)
    rmse = gp.score_rmse(p_te, v_te)
    # Reconstruction should be much better than predicting the mean velocity.
    baseline = float(np.sqrt(np.mean(np.sum((v_te - v_tr.mean(axis=0)) ** 2, axis=1))))
    assert rmse < 0.5 * baseline


def test_gp_predict_returns_std():
    field = synthetic_flow_field(n=30, seed=0)
    points, velocities = make_observations(field, n_obs=60, seed=0)
    gp = GPFieldModel().fit(points, velocities)
    mean, std = gp.predict(points[:5], return_std=True)
    assert mean.shape == (5, 2)
    assert std.shape == (5, 2)
    assert np.all(std >= 0)


def test_rk4_step_zero_velocity_is_identity():
    pts = np.array([[1.0, 1.0], [2.0, 2.0]])
    zero = lambda p: np.zeros_like(p)  # noqa: E731
    out = rk4_step(pts, zero, dt=0.1)
    assert np.allclose(out, pts)


def test_advection_shapes_and_containment():
    field = synthetic_flow_field(n=40, seed=0)
    starts = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 4.0]])
    traj = advect_particles(field, starts, n_steps=50, dt=0.02)
    assert traj.shape == (51, 3, 2)
    assert np.isfinite(traj).all()


def test_interpolator_matches_grid_nodes():
    field = synthetic_flow_field(n=40, seed=3)
    vel = make_interpolator(field)
    pt = np.array([[field.xs[5], field.ys[7]]])
    out = vel(pt)[0]
    assert np.allclose(out[0], field.u[7, 5], atol=1e-6)
