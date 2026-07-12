"""Offline quickstart: reconstruct the flow from the committed sample.

Reads ``data/sample_observations.csv`` (no network, no field regeneration), fits
the Gaussian process on a training split of the noisy measurements, and reports
accuracy and calibration on the held-out points against the noise-free ground
truth carried in the sample. Optionally writes the fitted model artifact and a
metrics file.

Usage:
    python examples/reconstruct_from_sample.py
    python examples/reconstruct_from_sample.py --save-model results/gp_flow_model.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from flowgp.gp import GPFieldModel
from flowgp.metrics import coverage, nlpd, rmse
from flowgp.persistence import save_model


def load_sample(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the sample CSV into points, noisy velocities, and true velocities."""
    rows: list[list[float]] = []
    with Path(path).open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            rows.append([float(v) for v in row])
    arr = np.asarray(rows, dtype=float)
    points = arr[:, 0:2]
    obs = arr[:, 2:4]
    true = arr[:, 4:6]
    return points, obs, true


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/sample_observations.csv")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--metrics-out", default=None)
    args = parser.parse_args()

    points, obs, true = load_sample(args.sample)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(points))
    n_test = max(1, int(round(args.test_fraction * len(points))))
    te, tr = perm[:n_test], perm[n_test:]

    gp = GPFieldModel(length_scale=args.length_scale, seed=args.seed).fit(points[tr], obs[tr])
    mean, std = gp.predict(points[te], return_std=True)

    test_rmse = rmse(mean, true[te])
    test_nlpd = nlpd(mean, std, true[te])
    cov1 = coverage(mean, std, true[te], k=1.0)
    cov2 = coverage(mean, std, true[te], k=2.0)
    lml = gp.log_marginal_likelihood()

    print(f"Sample: {len(points)} observations, {len(tr)} train / {len(te)} test")
    print(f"Held-out RMSE vs ground truth : {test_rmse:.4f}")
    print(f"Held-out mean NLPD            : {test_nlpd:.4f}")
    print(f"1-sigma coverage (nominal .68): {cov1:.3f}")
    print(f"2-sigma coverage (nominal .95): {cov2:.3f}")
    print(f"Log marginal likelihood (u+v) : {lml:.2f}")

    if args.save_model:
        path = save_model(gp, args.save_model)
        print(f"Saved fitted model to {path}")

    if args.metrics_out:
        metrics = {
            "n_observations": int(len(points)),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "test_rmse": round(test_rmse, 6),
            "test_nlpd": round(test_nlpd, 6),
            "coverage_1sigma": round(cov1, 4),
            "coverage_2sigma": round(cov2, 4),
            "log_marginal_likelihood": round(lml, 4),
        }
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote metrics to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
