"""Save and load a fitted GP flow model as a small artifact.

The fitted :class:`~flowgp.gp.GPFieldModel` stores its training points and the
optimized kernel, so it can be serialized once and reloaded for instant
inference without refitting. The artifact is a few kilobytes for the sample size
used here.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from flowgp.gp import GPFieldModel


def save_model(model: GPFieldModel, path: str | Path) -> Path:
    """Serialize a fitted model to ``path`` and return the path.

    Args:
        model: A fitted :class:`~flowgp.gp.GPFieldModel`.
        path: Destination file (``.joblib`` by convention).

    Raises:
        RuntimeError: If the model has not been fit.
    """
    if not model.fitted:
        raise RuntimeError("refusing to save an unfitted GPFieldModel")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> GPFieldModel:
    """Load a model previously written by :func:`save_model`."""
    return joblib.load(Path(path))
