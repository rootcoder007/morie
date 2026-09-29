# morie.fn -- function file (rootcoder007/morie)
"""Spatial count bootstrap CI."""

from __future__ import annotations

import math

from . import _gravity as gv
from ._containers import SpatialResult
from ._qpcore import inverse
from ._rng import random_uniform
from .spcount import sar_poisson


def _rpois(mu, u):
    """Poisson quantile at ``u`` by summing the pmf from 0."""
    k, p = 0, math.exp(-mu)
    c = p
    while c < u and k < 100000:
        k += 1
        p *= mu / k
        c += p
    return k


def _polish(y, X, W, rho, lo=-0.99, hi=0.99):
    """Refine the spatial-lag Poisson ``rho`` by Newton steps on its analytic profile score.

    At ``rho`` the Poisson GLM on ``Z = (I - rho W)^{-1} X`` gives ``beta``
    and ``mu``; by the envelope theorem the profile score is ``(y - mu)'
    (I - rho W)^{-1} W Z beta``. Golden-section search leaves ``rho`` at
    about 1e-9; the Newton steps take it to machine precision.
    """
    n = len(y)

    def fit(r):
        A = [[(1.0 if i == j else 0.0) - r * W[i][j] for j in range(n)] for i in range(n)]
        Ai = [[float(v) for v in row] for row in inverse(A)]
        Z = [[gv.ssum(Ai[i][m] * X[m][k] for m in range(n)) for k in range(len(X[0]))] for i in range(n)]
        g = gv.glm_fit(Z, y, "poisson")
        eta = [gv.ssum(a * b for a, b in zip(row, g["coefficients"])) for row in Z]
        d = [gv.ssum(Ai[i][m] * gv.ssum(W[m][k] * eta[k] for k in range(n)) for m in range(n)) for i in range(n)]
        return gv.ssum((a - m) * t for a, m, t in zip(y, g["fitted"], d)), g

    for _ in range(6):
        s0, g = fit(rho)
        h = 1e-6
        ds = (fit(rho + h)[0] - fit(rho - h)[0]) / (2 * h)
        if not ds < 0.0:
            break
        new = rho - s0 / ds
        if not lo < new < hi:
            break
        step, rho = abs(new - rho), new
        if step < 1e-15:
            break
    s0, g = fit(rho)
    return rho, g


def _fit(y, X, W):
    f = sar_poisson(y, X, W)
    return _polish([float(v) for v in y], [[float(v) for v in r] for r in X], [[float(v) for v in r] for r in W], f.rho)


def scpboot(y, X, W, B=9, seed=0, level=0.95):
    r"""Spatial count bootstrap CI.

    Parametric bootstrap of the spatial-lag Poisson model ``E y = exp((I -
    rho W)^{-1} X beta)`` (:func:`morie.fn.spcount.sar_poisson`, Lambert,
    Brown and Florax 2010; ``rho`` refined by Newton steps on the analytic
    profile score): draw ``y* ~ Poisson(mu_hat)`` by inversion of
    Philox uniforms, refit, and take the type-7 quantiles ``(1 - level) / 2``
    and ``(1 + level) / 2`` of the refitted ``rho`` (Efron and Tibshirani
    1993, ch. 6).

    Parameters
    ----------
    y : array-like, shape (n,)
        Counts.
    X : array-like, shape (n, p)
        Design matrix with intercept.
    W : array-like, shape (n, n)
        Spatial weights.
    B : int
        Bootstrap replicates.
    seed : int
        Philox seed.
    level : float
        Confidence level.

    Returns
    -------
    SpatialResult
        ``statistic`` is the ML ``rho``; ``extra`` has ``ci_lower``,
        ``ci_upper``, ``se_boot`` and ``draws``.

    References
    ----------
    Lambert, D. M., Brown, J. P. and Florax, R. J. G. M. (2010). A two-step estimator for a spatial lag
    model of counts. *Regional Science and Urban Economics*, 40(4), 241-252.

    Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> X = [[1, 0.1], [1, 0.4], [1, 0.5], [1, 0.9]]
    >>> r = scpboot([1, 3, 4, 7], X, W, B=4, seed=1)
    >>> round(r.statistic, 6), len(r.extra["draws"])
    (0.358854, 4)
    """
    rho, g = _fit(y, X, W)
    mu = list(g["fitted"])
    n = len(mu)
    u = random_uniform(int(B) * n, seed=seed)
    u = [float(v) for v in (u.tolist() if hasattr(u, "tolist") else u)]
    draws = []
    for b in range(int(B)):
        ys = [_rpois(mu[i], u[b * n + i]) for i in range(n)]
        draws.append(_fit(ys, X, W)[0])
    s = sorted(draws)

    def q7(q):
        h = (len(s) - 1) * q
        lo = math.floor(h)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (h - lo) * (s[hi] - s[lo])

    a = (1.0 - float(level)) / 2.0
    m = math.fsum(draws) / len(draws)
    sd = math.sqrt(math.fsum((v - m) ** 2 for v in draws) / (len(draws) - 1)) if len(draws) > 1 else math.nan
    return SpatialResult(
        name="scpboot",
        statistic=rho,
        extra={"ci_lower": q7(a), "ci_upper": q7(1.0 - a), "se_boot": sd, "draws": draws},
    )


scpboot_fn = scpboot


def cheatsheet() -> str:
    return "scpboot(y, X, W, B, seed, level) -> parametric-bootstrap CI for the SAR-Poisson rho"
