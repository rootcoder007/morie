# morie.fn -- function file (rootcoder007/morie)
"""Stratified mean and its variance."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["stratmean", "stratified_mean"]


def stratmean(y, h, Nh, level=0.95):
    """Stratified estimate of the population mean, with its variance.

    The whole point of Cochran's Theorem 5.3 is in the cross terms
    that are not there: because the strata are sampled independently,
    the variance of ybar_st is a weighted sum of WITHIN-stratum
    variances only, so stratification can only help.  Note that
    ybar_st is not in general the sample mean -- it is only when the
    allocation is proportional -- which is why the unweighted mean is
    returned alongside for contrast.

    ``h`` holds ONE-BASED stratum labels 1..L, matching the R arm.

    Formula: W_h = N_h/N,  ybar_st = sum_h W_h ybar_h,
             V(ybar_st) = sum_h W_h^2 (1 - f_h) s_h^2 / n_h,
             f_h = n_h/N_h

    Parameters
    ----------
    y : array-like
        Observations.
    h : array-like of int
        One-based stratum label of each observation.
    Nh : array-like
        Population size of each stratum, length L.
    level : float
        Confidence level for the returned interval.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``stratum_mean``, ``stratum_var``, ``nh``, ``Wh``,
        ``unweighted_mean``, ``N``, ``n``, ``L``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Section 5.3,
    Theorem 5.3: V(ybar_st) = sum W_h^2 S_h^2 (1 - f_h)/n_h, the cross
    terms vanishing because the strata are drawn independently.
    Chapter 5 read from the scanned original.  Cross-checked against
    the reference implementation in the CRAN package ``samplingbook``
    1.2.4, whose ``stratamean`` forms ``sum(Meanh*wh)`` and
    ``sum(Varh*wh^2)`` with ``Varh`` the finite-population-corrected
    within-stratum variance.
    """
    y = C.vec(y)
    h = [int(v) for v in C.vec(h)]
    Nh = C.vec(Nh)
    if len(y) != len(h):
        raise ValueError("y and h must have the same length")
    L = len(Nh)
    if L < 1:
        raise ValueError("at least one stratum is required")
    if any(not 1 <= v <= L for v in h):
        raise ValueError("h must hold one-based stratum labels in 1..L")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    N = sum(Nh)
    W = [v / N for v in Nh]
    mh = []
    vh = []
    nh = []
    for s in range(1, L + 1):
        ys = [y[i] for i in range(len(y)) if h[i] == s]
        m = len(ys)
        if m < 2:
            raise ValueError("every stratum needs at least two observations")
        if m > Nh[s - 1]:
            raise ValueError("a stratum sample cannot exceed its population")
        nh.append(m)
        mh.append(sum(ys) / m)
        s2 = C.var(ys, 1)
        vh.append((Nh[s - 1] - m) / Nh[s - 1] * s2 / m)
    est = sum(W[i] * mh[i] for i in range(L))
    var = sum(W[i] * W[i] * vh[i] for i in range(L))
    se = math.sqrt(var)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    return RichResult(payload={
        "estimate": est, "se": se, "ci_lower": est - z * se,
        "ci_upper": est + z * se, "stratum_mean": mh, "stratum_var": vh,
        "nh": nh, "Wh": W, "unweighted_mean": sum(y) / len(y),
        "N": N, "n": len(y), "L": L,
        "method": "Stratified mean, Cochran Theorem 5.3"})


stratified_mean = stratmean


def cheatsheet():
    return "stratm: ybar_st = sum W_h ybar_h; V = sum W_h^2 (1-f_h) s_h^2/n_h"
