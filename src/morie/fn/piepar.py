# morie.fn -- function file (rootcoder007/morie)
"""Population intervention effect."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["pie_parameters"]


def pie_parameters(y, X, intervention_dist):
    """Effect of shifting the exposure distribution, not eliminating it.

    A total effect compares everyone exposed with everyone unexposed,
    which is often a comparison no policy could bring about.  The
    population intervention effect compares what happened with what
    would happen under a realistic redistribution of exposure, so the
    contrast is between the observed world and an attainable one.

    Formula: ``PIE = E[Y(do(X = x*))] - E[Y]``, with the first term
    obtained by standardising the fitted outcome regression over the
    proposed exposure distribution.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n, p)
        Exposure in the first column, covariates after it.
    intervention_dist : array-like
        Exposure values defining the intervened distribution; each is
        applied to every unit and the results averaged.

    Returns
    -------
    RichResult
        ``estimate`` (PIE), ``observed``, ``intervened``, ``se``, ``n``.

    References
    ----------
    Westreich, D. (2014).  From patients to policy: population
    intervention effects in epidemiology.  Epidemiology 25:437-440.
    """
    yv = C.vec(y)
    Xm = C.mat(X)
    n = len(yv)
    W = C.cbind1(Xm)
    b, fitted, resid, _ = S.ols(W, yv)
    xs = C.vec(intervention_dist)
    tot = 0.0
    for xv in xs:
        rows = [[1.0, xv] + list(Xm[i][1:]) for i in range(n)]
        tot += sum(C.dot(r, b) for r in rows) / n
    interv = tot / len(xs)
    obs = sum(yv) / n
    m = sum(resid) / n
    se = math.sqrt(sum((t - m) ** 2 for t in resid) / (n - 1) / n) if n > 1 else float("nan")
    return RichResult(payload={
        "estimate": interv - obs, "observed": obs, "intervened": interv,
        "se": se, "n": n,
        "method": "Population intervention effect"})


pieparameters = pie_parameters


def cheatsheet():
    return "piepar: Population intervention effect."
