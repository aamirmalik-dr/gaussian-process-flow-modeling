"""Gaussian process regression of a velocity field from sparse samples."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


class GPFieldModel:
    """Reconstruct a 2D velocity field with two Gaussian process regressors.

    The x- and y-velocity components are modeled independently, each with an RBF
    kernel plus a white-noise term. Predictions return both the posterior mean
    and standard deviation, so the reconstruction carries an uncertainty map.
    """

    def __init__(self, length_scale: float = 1.0, seed: int = 0) -> None:
        kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(0.1)
        self._gp_u = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
        self._gp_v = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
        self.fitted = False

    def fit(self, points: np.ndarray, velocities: np.ndarray) -> GPFieldModel:
        """Fit the model on observed points and velocities.

        Args:
            points: Observation locations, shape ``(m, 2)``.
            velocities: Observed velocities, shape ``(m, 2)``.
        """
        self._gp_u.fit(points, velocities[:, 0])
        self._gp_v.fit(points, velocities[:, 1])
        self.fitted = True
        return self

    def predict(self, points: np.ndarray, return_std: bool = False):
        """Predict velocities at ``points``.

        Returns:
            Either the mean velocities of shape ``(m, 2)``, or a tuple
            ``(mean, std)`` where ``std`` has shape ``(m, 2)`` if
            ``return_std`` is set.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.fitted:
            raise RuntimeError("GPFieldModel must be fit before predict")
        if return_std:
            mu_u, sd_u = self._gp_u.predict(points, return_std=True)
            mu_v, sd_v = self._gp_v.predict(points, return_std=True)
            return np.stack([mu_u, mu_v], axis=1), np.stack([sd_u, sd_v], axis=1)
        mu_u = self._gp_u.predict(points)
        mu_v = self._gp_v.predict(points)
        return np.stack([mu_u, mu_v], axis=1)

    def score_rmse(self, points: np.ndarray, velocities: np.ndarray) -> float:
        """Root mean squared error of the predicted velocity vectors."""
        pred = self.predict(points)
        return float(np.sqrt(np.mean(np.sum((pred - velocities) ** 2, axis=1))))

    def log_marginal_likelihood(self) -> float:
        """Sum of the two component log marginal likelihoods at the fitted kernel.

        The marginal likelihood is what the GP maximizes when it tunes its kernel
        hyperparameters, so it summarizes how well the fitted model explains the
        training data. The two velocity components are independent regressors, so
        their log marginal likelihoods add.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self.fitted:
            raise RuntimeError("GPFieldModel must be fit before log_marginal_likelihood")
        return float(
            self._gp_u.log_marginal_likelihood_value_ + self._gp_v.log_marginal_likelihood_value_
        )

    @property
    def kernel_(self) -> tuple:
        """The two fitted kernels ``(u_kernel, v_kernel)`` after optimization."""
        return self._gp_u.kernel_, self._gp_v.kernel_
