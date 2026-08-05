# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric (kernel) worst-case regression bound."""

import math

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_nonparam_regr"]


def bound_nonparam_regr(y, D, X, bw):
    """Worst-case ATE bounds computed conditional on a covariate.

    The worst-case decomposition is applied at each covariate value with
    Nadaraya-Watson estimates of ``E(y | X = x, D = t)`` and
    ``P(D = t | X = x)``, and the resulting pointwise bounds are averaged
    over the empirical distribution of ``X``.  Conditioning can only help:
    as the bandwidth grows the estimator collapses to the unconditional
    worst-case bound, and for any finite bandwidth the averaged interval is
    no wider.

    Formula: ``E_X [ m_1(X) p_1(X) + y_0 (1 - p_1(X))
                     - m_0(X) p_0(X) - y_1 (1 - p_0(X)) ]`` for the lower
    bound and the mirror expression for the upper, with Gaussian kernel
    weights ``exp(-((x_i - x_j) / h)^2 / 2)``.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment indicator, coded 0/1.
    X : array-like
        Scalar conditioning covariate, one value per unit.
    bw : float
        Kernel bandwidth, strictly positive.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``bw``, ``n``.

    References
    ----------
    Manski, C. F. (2003).  Partial Identification of Probability
    Distributions.  Springer, New York.  The conditional worst-case bound
    evaluated here is equation (2.11) of Molinari, F. (2021),
    Microeconometrics with partial identification, Handbook of Econometrics
    7A (arXiv:2004.11751 p. 17), applied at each ``x``.
    """
    yv, dv = B.yd(y, D, "bound_nonparam_regr")
    xv = C.vec(X)
    n = len(yv)
    if len(xv) != n:
        raise ValueError("bound_nonparam_regr: X must have one value per unit")
    h = float(bw)
    if not (h > 0.0):
        raise ValueError("bound_nonparam_regr: bw must be positive")
    y0, y1 = B.support(yv)
    slo = 0.0
    shi = 0.0
    for i in range(n):
        w1 = 0.0
        w0 = 0.0
        s1 = 0.0
        s0 = 0.0
        for j in range(n):
            u = (xv[i] - xv[j]) / h
            k = math.exp(-0.5 * u * u)
            if dv[j] == 1.0:
                w1 += k
                s1 += k * yv[j]
            else:
                w0 += k
                s0 += k * yv[j]
        wt = w1 + w0
        p1 = w1 / wt
        p0 = w0 / wt
        m1 = s1 / w1 if w1 > 0.0 else 0.0
        m0 = s0 / w0 if w0 > 0.0 else 0.0
        a1 = B.wc_arm(m1, p1, y0, y1)
        a0 = B.wc_arm(m0, p0, y0, y1)
        slo += a1[0] - a0[1]
        shi += a1[1] - a0[0]
    lo = slo / n
    hi = shi / n
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "bw": h, "n": n,
        "method": "Nonparametric regression bound"})


def cheatsheet():
    return "bndnpr: Nonparametric regression bound"
