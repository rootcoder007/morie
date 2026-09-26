"""Bayesian kriging with the Matern class, Handcock and Stein (1993)."""

import math

from . import _array_core as np
from ._richresult import RichResult
from .bkrnig import bayesian_kriging_nig
from .spmatr import schabenberger_matern_covariance

__all__ = ["bayesian_kriging_matern"]


def _corr(D, nu, theta):
    flat = [float(v) for row in D for v in row]
    c = schabenberger_matern_covariance(flat, 1.0, nu, theta)["covariance"]
    k = len(D[0])
    return [[float(c[i * k + j]) for j in range(k)] for i in range(len(D))]


def bayesian_kriging_matern(coords, z, X, coords0, X0, theta_grid, nu_grid, prior_weights=None):
    r"""Bayesian kriging over the Matern class with prior :math:`\pi(\theta)/\sigma^2`.

    Handcock & Stein (1993), as described by Schabenberger & Gotway (2005,
    Sec. 6.4.3.1, p. 394): :math:`Z(s) \sim G(X\beta, \sigma^2 V(\theta))`
    with V the Matern correlation (4.9) in scale :math:`\theta` and
    smoothness :math:`\nu`, prior :math:`\pi(\beta, \sigma^2, \theta) \propto
    \pi(\theta)/\sigma^2`, and bounded uniform priors on :math:`\theta` and
    :math:`\nu`. Integrating out :math:`\beta` and :math:`\sigma^2` leaves

    .. math::

        p(\theta, \nu \mid Z) \propto \pi(\theta, \nu)\,|V|^{-1/2}
        |X'V^{-1}X|^{-1/2}\,(S^2)^{-(n-p)/2},

    :math:`S^2` the GLS residual sum of squares, and at each
    :math:`(\theta, \nu)` the predictive is the flat-prior t of (6.97)-(6.98)
    on :math:`n - p` degrees of freedom. There is no closed form over
    :math:`(\theta, \nu)`; this evaluates the posterior on the given grids
    (a quadrature rule for the uniform priors) and returns the mixture's mean
    and variance. Bounded uniform priors keep the posterior proper, which the
    unbounded noninformative choices need not be (Berger, De Oliveira & Sanso
    2001).

    Parameters
    ----------
    coords, coords0 : (n, 2), (n0, 2) array-like
    z : (n,) array-like
    X, X0 : (n, p), (n0, p) array-like
    theta_grid, nu_grid : sequence of float
        Grid points of the uniform priors on the scale and smoothness.
    prior_weights : (len(theta_grid), len(nu_grid)) array-like, optional
        Quadrature weights of the prior on the grid, default equal.

    Returns
    -------
    RichResult
        ``mean``, ``variance`` (predictive, one per location),
        ``posterior`` (grid of posterior probabilities), ``theta_grid``,
        ``nu_grid``.

    References
    ----------
    Handcock, M. S. & Stein, M. L. (1993). A Bayesian analysis of kriging.
    Technometrics 35, 403-410. Berger, J. O., De Oliveira, V. & Sanso, B.
    (2001). Objective Bayesian analysis of spatially correlated data. JASA
    96, 1361-1374. Schabenberger, O. & Gotway, C. A. (2005). Statistical
    Methods for Spatial Data Analysis. Chapman & Hall/CRC, p. 394.
    """
    pts = [tuple(float(v) for v in p) for p in coords]
    p0 = [tuple(float(v) for v in p) for p in coords0]
    zz = [float(v) for v in z]
    Xm = [[float(v) for v in row] for row in X]
    X0m = [[float(v) for v in row] for row in X0]
    n, p = len(zz), len(Xm[0])
    if n - p <= 2:
        raise ValueError("need n - p > 2")
    D = [[math.dist(a, b) for b in pts] for a in pts]
    D0 = [[math.dist(a, b) for b in pts] for a in p0]
    D00 = [[math.dist(a, b) for b in p0] for a in p0]
    tg, ng = [float(v) for v in theta_grid], [float(v) for v in nu_grid]
    pw = [[1.0] * len(ng) for _ in tg] if prior_weights is None else [[float(v) for v in r] for r in prior_weights]
    logw, preds = [], []
    for i, th in enumerate(tg):
        for j, nu in enumerate(ng):
            V = _corr(D, nu, th)
            Va = np.asarray(V)
            fit = bayesian_kriging_nig(Xm, zz, X0m, V, _corr(D0, nu, th), _corr(D00, nu, th))
            S2 = fit["a_star"]
            A = np.asarray(Xm).T @ np.linalg.solve(Va, np.asarray(Xm))
            ld = float(np.linalg.slogdet(Va)[1]) + float(np.linalg.slogdet(A)[1])
            logw.append(math.log(pw[i][j]) - 0.5 * ld - 0.5 * (n - p) * math.log(S2))
            var = fit["variance"]
            preds.append((fit["mean"], [float(var[k][k]) for k in range(len(p0))]))
    top = max(logw)
    w = [math.exp(v - top) for v in logw]
    tot = sum(w)
    w = [v / tot for v in w]
    n0 = len(p0)
    mean = [sum(wi * pr[0][k] for wi, pr in zip(w, preds)) for k in range(n0)]
    var = [sum(wi * (pr[1][k] + pr[0][k] ** 2) for wi, pr in zip(w, preds)) - mean[k] ** 2 for k in range(n0)]
    post = [w[i * len(ng) : (i + 1) * len(ng)] for i in range(len(tg))]
    return RichResult(
        title="Bayesian kriging over the Matern class (Handcock and Stein 1993)",
        summary_lines=[("grid", f"{len(tg)} x {len(ng)}"), ("locations", n0)],
        payload={"mean": mean, "variance": var, "posterior": post, "theta_grid": tg, "nu_grid": ng},
    )


def cheatsheet():
    return "hsbkrg: Handcock-Stein Bayesian kriging, posterior over Matern (theta, nu) on a grid"
