# morie.fn -- function file (rootcoder007/morie)
r"""INLA: deterministic posteriors for latent Gaussian models.

A large family of models -- generalised linear, additive, spatial,
spatio-temporal, geoadditive -- share one structure: a **latent
Gaussian field** :math:`x` controlled by a **few** hyperparameters
:math:`\theta`, observed through a non-Gaussian likelihood. Because
the response is non-Gaussian the posterior marginals have no closed
form, and the default answer was MCMC.

**The objection to MCMC here is practical, not philosophical.** For
these models it has problems of convergence *and* of computational
time, to the point that in some applications it is simply not an
appropriate tool for routine analysis. INLA computes very accurate
approximations to the posterior marginals **deterministically** --
seconds or minutes where MCMC needs hours or days -- and the same
machinery gives model comparison criteria and predictive measures, so
models can be compared automatically.

**The approximation is nested, which is what "nested" means here.**

.. math:: \tilde p(x_i \mid y) = \int \tilde p(x_i \mid \theta, y)\,
          \tilde p(\theta \mid y)\, d\theta,

where the inner conditional is itself a Laplace approximation and the
outer integral is a **finite weighted sum** over a small design of
:math:`\theta` points. That the sum is small is exactly why
:math:`\dim(\theta)` must stay low -- the assumption is structural,
not incidental, and ``integrate_marginals`` refuses a design that
implies otherwise.

**The Gaussian approximation is the cheap inner step**: match the mode
and the curvature of :math:`\log p(x\mid\theta,y)`, which for a
log-concave likelihood is a Newton iteration on a sparse system. The
**simplified Laplace** correction adds the skewness the Gaussian
misses, and the anchor exercises exactly that -- on a Gaussian
likelihood the approximation must be **exact**, and on a skewed one it
must not be.

References
----------
Rue, H., Martino, S. & Chopin, N. (2009) "Approximate Bayesian
inference for latent Gaussian models by using integrated nested
Laplace approximations", *Journal of the Royal Statistical Society:
Series B (Statistical Methodology)* 71(2), 319-392,
doi:10.1111/j.1467-9868.2008.00700.x. [PDF supplied by Vee.] The
abstract and Sec. 1: approximate Bayesian inference for latent
Gaussian models, a subset of structured additive regression models in
which the latent field is Gaussian, controlled by a FEW
hyperparameters and observed with non-Gaussian response variables, so
that the posterior marginals are not available in closed form; that
MCMC can be implemented but is not without problems of convergence and
computational time, to the extent that in some applications it is
simply not an appropriate tool for routine analysis; the integrated
nested Laplace approximation and its simplified version giving very
accurate approximations to the posterior marginals directly; the
computational benefit of seconds or minutes against hours or days; and
the generality allowing automatic, streamlined Bayesian analysis with
model comparison criteria and predictive measures.

Tierney, L. & Kadane, J. B. (1986) "Accurate Approximations for
Posterior Moments and Marginal Densities", *Journal of the American
Statistical Association* 81(393), 82-86,
doi:10.1080/01621459.1986.10478240. The Laplace approximation being
nested.

Rue, H. & Held, L. (2005) *Gaussian Markov Random Fields: Theory and
Applications*, Monographs on Statistics and Applied Probability 104,
Chapman & Hall/CRC, doi:10.1201/9780203492024. The sparse GMRF
computations the method rests on.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["gaussian_approximation", "laplace_marginal",
           "hyperparameter_design", "integrate_marginals",
           "skewness_correction"]

_EPS = 1e-12
_MAX_HYPER = 6


def gaussian_approximation(log_lik, log_lik_d1, log_lik_d2, prior_mean,
                           prior_precision, x0=0.0, iters=60,
                           tol=1e-12):
    r"""Match mode and curvature of :math:`\log p(x\mid\theta,y)`.

    Newton on a log-concave objective: the second derivative of the
    likelihood adds to the prior precision, which is the sparse
    update INLA performs on the whole field at once.
    """
    m, Q = float(prior_mean), float(prior_precision)
    if Q <= 0.0:
        raise ValueError("inlasm: the prior precision must be "
                         "positive")
    x = float(x0)
    it = 0
    for it in range(1, int(iters) + 1):
        g = float(log_lik_d1(x)) - Q * (x - m)
        h = float(log_lik_d2(x)) - Q
        if h >= -_EPS:
            raise ValueError("inlasm: the objective is not locally "
                             "concave at x = %r, so the Gaussian "
                             "approximation has no mode here" % (x,))
        step = g / h
        x -= step
        if abs(step) < float(tol):
            break
    prec = Q - float(log_lik_d2(x))
    return {"mode": x, "precision": prec,
            "sd": 1.0 / math.sqrt(prec), "iterations": it,
            "log_norm": float(log_lik(x)) - 0.5 * Q * (x - m) ** 2
            + 0.5 * math.log(2.0 * math.pi / prec),
            "note": "exact when the likelihood is Gaussian, because "
                    "then the objective is exactly quadratic"}


def skewness_correction(third_derivative, precision):
    r"""What the simplified Laplace adds over the Gaussian.

    A Gaussian is symmetric by construction, so the third derivative
    of the log density is the leading term it cannot represent.
    """
    d3, prec = float(third_derivative), float(precision)
    if prec <= 0.0:
        raise ValueError("inlasm: the precision must be positive")
    return {"skewness": d3 / prec ** 1.5,
            "gaussian_adequate": abs(d3 / prec ** 1.5) < 1e-9,
            "note": "zero for a Gaussian likelihood; non-zero is "
                    "exactly what the simplified Laplace corrects"}


def laplace_marginal(log_joint, x_grid, theta):
    r"""A marginal on a grid, by normalising the Laplace values."""
    xs = [float(v) for v in k.vec(x_grid)]
    if len(xs) < 2:
        raise ValueError("inlasm: at least 2 grid points are needed")
    lp = [float(log_joint(x, theta)) for x in xs]
    m = max(lp)
    w = [math.exp(v - m) for v in lp]
    area = 0.0
    for i in range(len(xs) - 1):
        area += 0.5 * (w[i] + w[i + 1]) * (xs[i + 1] - xs[i])
    if area <= _EPS:
        raise ValueError("inlasm: the marginal has no mass on this "
                         "grid")
    dens = [v / area for v in w]
    mean = 0.0
    for i in range(len(xs) - 1):
        mean += 0.5 * (dens[i] * xs[i] + dens[i + 1] * xs[i + 1]) \
            * (xs[i + 1] - xs[i])
    var = 0.0
    for i in range(len(xs) - 1):
        var += 0.5 * (dens[i] * (xs[i] - mean) ** 2
                      + dens[i + 1] * (xs[i + 1] - mean) ** 2) \
            * (xs[i + 1] - xs[i])
    return {"x": xs, "density": dens, "mean": mean,
            "sd": math.sqrt(max(var, 0.0)), "log_scale": m}


def hyperparameter_design(mode, curvature, step=1.0, dim=None):
    r"""The small grid of :math:`\theta` points the outer sum runs over.

    A central composite design around the mode, scaled by the
    curvature. Its size is why :math:`\dim(\theta)` must stay small --
    the cost is exponential in it, and that is a stated restriction of
    the method, not a tuning choice.
    """
    m = [float(v) for v in k.vec(mode)]
    d = len(m) if dim is None else int(dim)
    if d != len(m):
        raise ValueError("inlasm: the mode has %d entries but dim is "
                         "%d" % (len(m), d))
    if d > _MAX_HYPER:
        raise ValueError("inlasm: %d hyperparameters -- the outer "
                         "integral is a small weighted SUM, so the "
                         "method assumes a low-dimensional theta "
                         "(the paper says a FEW)" % d)
    sd = [1.0 / math.sqrt(float(v)) if float(v) > 0.0 else 1.0
          for v in k.vec(curvature)]
    pts = [list(m)]
    for i in range(d):
        for s in (-1.0, 1.0):
            p = list(m)
            p[i] += s * float(step) * sd[i]
            pts.append(p)
    return {"points": pts, "n_points": len(pts), "dim": d,
            "cost_scaling": "linear here, exponential for a full "
                            "grid -- hence 'a few' hyperparameters",
            "note": "a FINITE design, so the outer integral is a sum"}


def integrate_marginals(conditional_marginals, log_weights, x_grid):
    r""":math:`\tilde p(x_i\mid y) = \sum_j \tilde p(x_i\mid\theta_j,y)
    \,\Delta_j`.

    The outer, "integrated" step: a weighted sum over the design,
    with the weights from :math:`\tilde p(\theta\mid y)`.
    """
    M = [[float(v) for v in k.vec(m)] for m in conditional_marginals]
    lw = [float(v) for v in k.vec(log_weights)]
    xs = [float(v) for v in k.vec(x_grid)]
    if len(M) != len(lw):
        raise ValueError("inlasm: %d conditional marginals but %d "
                         "weights" % (len(M), len(lw)))
    if any(len(m) != len(xs) for m in M):
        raise ValueError("inlasm: a conditional marginal does not "
                         "match the grid")
    mx = max(lw)
    w = [math.exp(v - mx) for v in lw]
    z = sum(w)
    w = [v / z for v in w]
    dens = [sum(w[j] * M[j][i] for j in range(len(M)))
            for i in range(len(xs))]
    area = 0.0
    for i in range(len(xs) - 1):
        area += 0.5 * (dens[i] + dens[i + 1]) * (xs[i + 1] - xs[i])
    if area > _EPS:
        dens = [v / area for v in dens]
    mean = 0.0
    for i in range(len(xs) - 1):
        mean += 0.5 * (dens[i] * xs[i] + dens[i + 1] * xs[i + 1]) \
            * (xs[i + 1] - xs[i])
    var = 0.0
    for i in range(len(xs) - 1):
        var += 0.5 * (dens[i] * (xs[i] - mean) ** 2
                      + dens[i + 1] * (xs[i + 1] - mean) ** 2) \
            * (xs[i + 1] - xs[i])
    return RichResult(payload={
        "estimate": mean, "mean": mean,
        "sd": math.sqrt(max(var, 0.0)), "density": dens, "x": xs,
        "theta_weights": w, "n_theta": len(M),
        "method": "integrated nested Laplace approximation; Rue, "
                  "Martino & Chopin (2009)",
        "note": "deterministic: no chain, no convergence diagnostic, "
                "and the same machinery yields model comparison and "
                "predictive measures",
    })


def cheatsheet():
    return ("inlasm: latent GAUSSIAN field x, a FEW hyperparameters "
            "theta, non-Gaussian response -- so the posterior marginals "
            "have no closed form. MCMC works in principle but has "
            "convergence AND time problems, sometimes badly enough "
            "that it is not appropriate for routine analysis. INLA is "
            "deterministic: p(x_i|y) = INTEGRAL p(x_i|theta,y) "
            "p(theta|y) dtheta, where the inner term is a LAPLACE "
            "approximation and the outer integral is a finite weighted "
            "SUM over a small design of theta -- which is exactly why "
            "dim(theta) must stay low. The Gaussian inner step is "
            "EXACT for a Gaussian likelihood; the simplified Laplace "
            "adds the skewness it cannot represent. Seconds or "
            "minutes against hours or days.")


# compact alias per ledger/NAMING.md
inla = integrate_marginals

# public names resolved by fn/_lazy_map.json
inla_spatial = integrate_marginals
inlaspatial = integrate_marginals
