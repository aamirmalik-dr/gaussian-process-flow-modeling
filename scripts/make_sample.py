"""Generate the committed sample of noisy flow observations.

The sample is a small carve of the synthetic divergence-free field: a few hundred
scattered points, each with the noisy measured velocity and the noise-free
analytic velocity at the same location. Committing both lets the offline
quickstart fit on the noisy measurements and still score accuracy and calibration
against ground truth, with no field regeneration and no network.

Usage:
    python scripts/make_sample.py --n-obs 300 --out data/sample_observations.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from flowgp.field import sample_velocity, synthetic_flow_field


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--grid", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/sample_observations.csv")
    args = parser.parse_args()

    field = synthetic_flow_field(n=args.grid, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    x_lo, x_hi = field.xs[0], field.xs[-1]
    y_lo, y_hi = field.ys[0], field.ys[-1]
    points = np.column_stack(
        [rng.uniform(x_lo, x_hi, args.n_obs), rng.uniform(y_lo, y_hi, args.n_obs)]
    )
    true_vel = sample_velocity(field, points)
    obs_vel = true_vel + rng.normal(0, args.noise, true_vel.shape)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "u_obs", "v_obs", "u_true", "v_true"])
        for (x, y), (uo, vo), (ut, vt) in zip(points, obs_vel, true_vel, strict=True):
            writer.writerow(
                [f"{x:.6f}", f"{y:.6f}", f"{uo:.6f}", f"{vo:.6f}", f"{ut:.6f}", f"{vt:.6f}"]
            )

    print(f"Wrote {args.n_obs} sample observations (noise={args.noise}) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
