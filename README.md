# Gaussian process flow modeling

Gaussian process regression and particle advection over a 2D velocity field, an
ocean-current-style spatiotemporal modeling demo that is fully reproducible from
a synthetic field with no external dataset.

The field is constructed from a stream function, so its velocity is the analytic
curl and the flow is divergence free (incompressible) by construction. A Gaussian
process then reconstructs the field from sparse noisy samples, with a posterior
uncertainty map, and a fourth-order Runge-Kutta integrator advects particles
through the field.

## What it does

- Builds a divergence-free velocity field from a sum of stream-function modes
  (`field.py`), with an analytic off-grid sampler and a numerical divergence
  check.
- Reconstructs the two velocity components with independent Gaussian processes
  (RBF plus white-noise kernels), returning mean and standard deviation
  (`gp.py`).
- Advects particles with RK4 and bilinear velocity interpolation, with boundary
  clamping so drifting particles stay defined (`advection.py`).
- Reports reconstruction RMSE and the field's residual divergence, and writes a
  trajectory figure and an uncertainty map (`scripts/simulate.py`).

## What it does not do

- The field is synthetic. `scripts/download_data.py` documents how to substitute
  a real public surface-current product (OSCAR or HYCOM), but no live ocean
  archive is fetched, both to keep the demo reproducible and because those
  archives are large and access can be unreliable.
- The GP uses a stationary RBF kernel; it does not model anisotropy or
  non-stationarity in the flow.
- Advection is kinematic (particles follow the field); there is no dynamics
  feedback.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run

```bash
python scripts/simulate.py --n-obs 200 --grid 45
```

`notebooks/demo.ipynb` is a short executed walkthrough.

## Results

Synthetic field on a 45 by 45 grid, 200 observations at noise level 0.05, 70/30
train/test split, seed 0. Produced by `scripts/simulate.py` in this repository.

| Quantity | Value |
|----------|-------|
| Residual divergence (mean absolute, interior) | 1.46e-02 |
| Residual divergence as a fraction of the mean derivative magnitude | 0.5% |
| GP reconstruction RMSE, held-out points (140 train obs) | 0.2440 |
| GP reconstruction RMSE, full grid | 0.2765 |

The field's residual divergence is only about half a percent of the size of its
velocity gradients, confirming numerically what the stream-function construction
guarantees analytically: the flow is incompressible, and the small residual is
finite-difference error that shrinks as the grid is refined (a test checks this).
The Gaussian process reconstructs the field from a couple of hundred noisy
samples with an RMSE well below the spread of the velocities, and its predictive
standard deviation grows away from the observations, exactly where a
reconstruction should be least certain.

## Layout

```
src/flowgp/     field, gp, advection, data
scripts/        simulate.py, download_data.py
notebooks/      demo.ipynb (executed)
tests/          pytest suite: divergence, GP reconstruction, RK4, advection
data/           gitignored; see data/README.md
```

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## License

MIT, see [LICENSE](LICENSE).

## Author

Aamir Malik. [GitHub](https://github.com/aamirmalik-dr) ·
[LinkedIn](https://linkedin.com/in/dr-aamirmalik)
