# Method note

This note derives the two moving parts of the demo: the divergence-free velocity
field with the Gaussian process that reconstructs it, and the Runge-Kutta scheme
that advects particles through it. The aim is to make every number the code
prints traceable to a line of math.

## 1. A divergence-free velocity field from a stream function

An incompressible two-dimensional flow satisfies

```
du/dx + dv/dy = 0
```

The clean way to guarantee this is to define the velocity as the curl of a scalar
stream function `psi(x, y)`:

```
u =  d psi / d y
v = -d psi / d x
```

Then, because mixed partial derivatives commute,

```
du/dx + dv/dy = d^2 psi / (dx dy) - d^2 psi / (dy dx) = 0
```

so the field is divergence free by construction, not by numerical accident. The
implementation uses a sum of sinusoidal modes,

```
psi(x, y) = sum_k a_k sin(fx_k x + px_k) sin(fy_k y + py_k)
```

whose curl has a closed form, so velocities are sampled analytically at any point
with no finite differencing:

```
u =  sum_k a_k fy_k sin(fx_k x + px_k) cos(fy_k y + py_k)
v = -sum_k a_k fx_k cos(fx_k x + px_k) sin(fy_k y + py_k)
```

`flowgp.field.divergence` still computes `du/dx + dv/dy` numerically on the grid.
It is not zero, but it is small: in the committed run its mean absolute value in
the interior is about half a percent of the mean gradient magnitude, and a test
confirms it shrinks as the grid is refined. That residual is finite-difference
error in the check, not a real source or sink in the field.

## 2. Gaussian process reconstruction

We observe the velocity at `m` scattered points with additive noise and want the
field everywhere, with error bars. A Gaussian process places a prior over
functions and conditions it on the data. Each velocity component is modeled by
its own GP, since the components are treated as independent scalar fields over
the same 2D input.

### Kernel

The prior covariance between the function values at two points `p` and `q` is set
by the kernel. This project uses a constant-scaled radial basis function plus a
white-noise term:

```
k(p, q) = sigma_f^2 exp( -||p - q||^2 / (2 l^2) ) + sigma_n^2 delta(p, q)
```

The RBF part says nearby points have correlated velocities and the correlation
decays smoothly over a length scale `l`. The white-noise part `sigma_n^2` models
the measurement noise on the observations. The signal variance `sigma_f^2`, the
length scale `l`, and the noise level `sigma_n^2` are hyperparameters, tuned by
maximizing the log marginal likelihood (scikit-learn does this on `fit`).

### Posterior mean and variance

Collect the training inputs into `X` and targets into `y`. Write `K = k(X, X)`
for the training covariance, `k_*` for the covariance between a test point and
all training points, and `k_**` for the test point's prior variance. The
posterior at the test point is Gaussian with

```
mean     mu(x*)    = k_*^T K^{-1} y
variance sigma^2(x*) = k_** - k_*^T K^{-1} k_*
```

Two things are worth noting. The mean is a weighted average of the observed
values, so it interpolates the data smoothly. The variance does not depend on the
observed values at all, only on where the observations sit: it is small close to
training points and rises toward `k_**` in the gaps. That is exactly the behavior
the hero figure shows, and it is why the GP gives an honest uncertainty map
rather than just a point estimate.

### What the metrics measure

- RMSE compares the posterior mean against the noise-free ground truth on
  held-out points: pure accuracy.
- NLPD (negative log predictive density) scores the full predictive Gaussian, so
  it punishes both wrong means and dishonest error bars. Lower is better.
- k-sigma coverage is the fraction of held-out truths inside the k standard
  deviation band. For a calibrated Gaussian it should sit near 0.68 at k = 1 and
  0.95 at k = 2. On this small sample the coverage runs a little conservative
  (wider bands than strictly needed), which is the safe direction to err.

## 3. Particle advection with Runge-Kutta

Given a velocity field `V(p)`, a passive particle follows the ordinary
differential equation

```
dp/dt = V(p)
```

We integrate it with the classical fourth-order Runge-Kutta step. From position
`p_n` with step `dt`:

```
k1 = V(p_n)
k2 = V(p_n + (dt/2) k1)
k3 = V(p_n + (dt/2) k2)
k4 = V(p_n + dt k3)
p_{n+1} = p_n + (dt/6) (k1 + 2 k2 + 2 k3 + k4)
```

RK4 has local error `O(dt^5)` and global error `O(dt^4)`, far more accurate than a
forward Euler step for the same `dt`, which matters when a particle circulates
many times through a vortex and small per-step errors would otherwise accumulate.

`V` here is a bilinear interpolation of the gridded field, with out-of-domain
points clamped to the boundary so a drifting particle always has a defined
velocity. Because the same interface accepts any callable, the GP posterior mean
can be dropped in as `V` to advect particles through the reconstructed field
instead of the true one.

## References

- Rasmussen and Williams, Gaussian Processes for Machine Learning, MIT Press,
  2006 (kernels, posterior mean and variance, marginal likelihood).
- Press et al., Numerical Recipes (Runge-Kutta integration of ODEs).
