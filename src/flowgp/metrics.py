"""Reconstruction and calibration metrics for the GP flow model.

A Gaussian process gives more than a point estimate. Each prediction is a normal
distribution with a mean and a standard deviation, so the reconstruction can be
judged not only on how close the mean is (accuracy) but on whether the reported
uncertainty is honest (calibration). The helpers here quantify both.
"""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error of predicted velocity vectors.

    Args:
        pred: Predicted velocities, shape ``(m, 2)``.
        target: True velocities, shape ``(m, 2)``.

    Returns:
        The vector RMSE, ``sqrt(mean(|pred - target|^2))``.
    """
    return float(np.sqrt(np.mean(np.sum((pred - target) ** 2, axis=1))))


def nlpd(pred: np.ndarray, std: np.ndarray, target: np.ndarray) -> float:
    """Mean negative log predictive density under the GP posterior.

    Treats each velocity component as an independent Gaussian with the predicted
    mean and standard deviation, and averages the negative log density of the
    true value across all components. Lower is better. This rewards a model whose
    error bars are neither over-confident (large penalty when a true value sits
    many sigma from the mean) nor needlessly wide.

    Args:
        pred: Posterior mean velocities, shape ``(m, 2)``.
        std: Posterior standard deviations, shape ``(m, 2)``.
        target: True velocities, shape ``(m, 2)``.

    Returns:
        The mean NLPD over all ``2 * m`` scalar components.
    """
    var = np.clip(std, 1e-9, None) ** 2
    term = 0.5 * (np.log(2 * np.pi * var) + (target - pred) ** 2 / var)
    return float(np.mean(term))


def coverage(pred: np.ndarray, std: np.ndarray, target: np.ndarray, k: float = 2.0) -> float:
    """Fraction of true values inside the ``k``-sigma predictive interval.

    For a well-calibrated Gaussian posterior this should approach the nominal
    normal coverage: about 0.683 at ``k = 1`` and about 0.954 at ``k = 2``.

    Args:
        pred: Posterior mean velocities, shape ``(m, 2)``.
        std: Posterior standard deviations, shape ``(m, 2)``.
        target: True velocities, shape ``(m, 2)``.
        k: Half-width of the interval in standard deviations.

    Returns:
        The empirical coverage in ``[0, 1]`` over all scalar components.
    """
    inside = np.abs(target - pred) <= k * np.clip(std, 1e-9, None)
    return float(np.mean(inside))
