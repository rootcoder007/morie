# morie.fn -- function file (rootcoder007/morie)
"""Nakagawa-Schielzeth marginal and conditional R-squared."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["nakagawa_conditional_r2"]


def nakagawa_conditional_r2(y, X=None, Z=None, cluster=None):
    """
    Nakagawa-Schielzeth R-squared for a mixed model

    Formula: R2_c = (sigma2_f + sum sigma2_l) / (sigma2_f + sum sigma2_l + sigma2_e)

    The marginal R2 drops the random-effect variance from the
    numerator and keeps it in the denominator, so it measures what the
    fixed effects alone explain; the conditional R2 credits both.  A
    residual variance of exactly zero therefore forces R2_c = 1, which
    is the degenerate case used to check the algebra.  Variance
    components come from a one-way random-intercept fit by moments.

    Parameters
    ----------
    y : array-like
        Response.
    X : array-like or None
        Fixed-effect design (no intercept column needed).
    Z : array-like or None
        Ignored; the random effect is the intercept of ``cluster``.
    cluster : array-like or None
        Grouping factor.

    Returns
    -------
    result : dict
        Keys: estimate (R2_conditional), r2_marginal, r2_conditional,
        var_fixed, var_random, var_resid, icc, n, n_groups.

    References
    ----------
    Nakagawa & Schielzeth (2013), Methods Ecol. Evol. 4(2):133-142,
    doi:10.1111/j.2041-210X.2012.00261.x.
    """
    yv = core.vec(y)
    n = len(yv)
    if n < 3:
        raise ValueError("need at least three observations")
    if X is None:
        Xm = [[] for _ in range(n)]
    else:
        Xm = core.mat(X)
        if len(Xm) != n:
            raise ValueError("y and X must have the same number of rows")
    D = [[1.0] + list(r) for r in Xm]
    p = len(D[0])
    if n <= p:
        raise ValueError("need more observations than fixed effects")
    beta = core.cholsolve(core.crossprod(D), core.matvec(core.tr(D), yv))
    fit = core.matvec(D, beta)
    mf = sum(fit) / n
    var_f = sum((v - mf) ** 2 for v in fit) / (n - 1)
    res = [yv[i] - fit[i] for i in range(n)]
    if cluster is None:
        var_r = 0.0
        a = 1
        var_e = sum(v * v for v in res) / (n - p)
    else:
        ids = list(cluster)
        if len(ids) != n:
            raise ValueError("y and cluster must have the same length")
        keys = []
        for k in ids:
            if k not in keys:
                keys.append(k)
        a = len(keys)
        groups = [[res[i] for i in range(n) if ids[i] == k] for k in keys]
        sizes = [len(g) for g in groups]
        gm = sum(res) / n
        ssb = sum(sizes[j] * (sum(groups[j]) / sizes[j] - gm) ** 2
                  for j in range(a))
        ssw = sum((v - sum(g) / len(g)) ** 2 for g in groups for v in g)
        if a > 1 and n > a:
            msb = ssb / (a - 1)
            msw = ssw / (n - a)
            m0 = (n - sum(s * s for s in sizes) / float(n)) / (a - 1)
            var_e = msw
            var_r = max((msb - msw) / m0, 0.0)
        else:
            var_e = ssw / max(n - a, 1)
            var_r = 0.0
    tot = var_f + var_r + var_e
    if tot <= 0.0:
        raise ValueError("total variance is zero; R-squared is undefined")
    r2m = var_f / tot
    r2c = (var_f + var_r) / tot
    icc = var_r / (var_r + var_e) if (var_r + var_e) > 0.0 else float("nan")
    return RichResult(payload={
        "estimate": r2c,
        "r2_marginal": r2m,
        "r2_conditional": r2c,
        "var_fixed": var_f,
        "var_random": var_r,
        "var_resid": var_e,
        "icc": icc,
        "n": n,
        "n_groups": a,
        "method": "Nakagawa-Schielzeth marginal and conditional R-squared",
    })


def cheatsheet():
    return "ccngg: Nakagawa-Schielzeth conditional R-squared"


# compact alias per ledger/NAMING.md
nakagawaconditionalr2 = nakagawa_conditional_r2
