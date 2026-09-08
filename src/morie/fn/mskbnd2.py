# morie.fn -- function file (rootcoder007/morie)
"""Manski no-assumption bounds on the ATE, refined by outcome covariates."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["manski_no_assumption_outcome", "manskinoassumptionoutcome"]


def manski_no_assumption_outcome(y, D, X, y_min, y_max):
    """Worst-case ATE bounds computed within covariate strata.

    With no assumption beyond a known outcome support ``[ymin, ymax]``,
    each arm mean is identified only up to what the unobserved
    counterfactuals could be.  Inside a stratum ``X = x``:

        E[Y(1)|x] in [ m1 p1 + ymin (1-p1),  m1 p1 + ymax (1-p1) ]
        E[Y(0)|x] in [ m0 p0 + ymin (1-p0),  m0 p0 + ymax (1-p0) ]

    with ``p1 = P(D=1|x)``, ``p0 = 1 - p1``, ``m1 = E[Y|D=1,x]`` and
    ``m0 = E[Y|D=0,x]``.  Differencing the opposite ends gives the
    stratum ATE bound, whose width is exactly ``ymax - ymin`` however the
    data fall -- the signature of the no-assumption bound, and the reason
    it always contains zero.

    Two population summaries follow, and they answer different questions:

    ``lower``/``upper`` average the stratum bounds with weights ``P(x)``.
    That is the bound on the population ATE ``sum_x P(x) ATE(x)`` and is
    weakly tighter than pooling the data first.

    ``inter_lower``/``inter_upper`` INTERSECT the stratum bounds.  That
    is only valid under the extra assumption that the effect is common
    across strata, so it is reported separately and never merged into
    the first pair.  The intersection can never be empty, and no
    emptiness flag is returned: since ``m1`` lies in ``[ymin, ymax]``,
    every stratum lower bound is at most ``p0 (ymin - m0) <= 0`` and
    every stratum upper bound at least ``p1 (ymax - m1) >= 0``, so all of
    them straddle zero.  A field that can only ever take one value is
    not a decision and is not reported as one.

    Parameters
    ----------
    y : array-like
        Observed outcomes, all within ``[y_min, y_max]``.
    D : array-like
        Binary treatment indicator.
    X : array-like
        Stratum label per unit (any values compared by equality).
    y_min, y_max : float
        A priori outcome support.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``inter_lower``,
        ``inter_upper``, ``n_strata``, ``n``.

    References
    ----------
    Manski, C. F. (1990), "Nonparametric bounds on treatment effects",
    American Economic Review: Papers and Proceedings 80(2), 319-323.
    Manski, C. F. (2003), Partial Identification of Probability
    Distributions, Springer.  Standard published form; neither source was
    in the local corpus and the AER proceedings page was not retrievable,
    so the bound is stated in full above for checking.
    """
    yv = C.vec(y)
    d = C.vec(D)
    n = len(yv)
    if n == 0:
        raise ValueError("y is empty")
    xs = list(X)
    if len(d) != n or len(xs) != n:
        raise ValueError("y, D and X must have the same length")
    lo, hi = float(y_min), float(y_max)
    if lo > hi:
        raise ValueError("y_min must not exceed y_max")
    if any(v < lo or v > hi for v in yv):
        raise ValueError("observed outcomes must lie in [y_min, y_max]")
    if any(v != 0.0 and v != 1.0 for v in d):
        raise ValueError("D must be binary 0/1")

    levels = []
    for v in xs:
        if v not in levels:
            levels.append(v)
    tot_lo = 0.0
    tot_hi = 0.0
    ilo = float("-inf")
    ihi = float("inf")
    for lev in levels:
        idx = [i for i in range(n) if xs[i] == lev]
        nk = len(idx)
        t = [i for i in idx if d[i] == 1.0]
        c = [i for i in idx if d[i] == 0.0]
        p1 = len(t) / nk
        p0 = len(c) / nk
        m1 = sum(yv[i] for i in t) / len(t) if t else 0.0
        m0 = sum(yv[i] for i in c) / len(c) if c else 0.0
        e1lo = m1 * p1 + lo * p0
        e1hi = m1 * p1 + hi * p0
        e0lo = m0 * p0 + lo * p1
        e0hi = m0 * p0 + hi * p1
        klo = e1lo - e0hi
        khi = e1hi - e0lo
        w = nk / n
        tot_lo += w * klo
        tot_hi += w * khi
        if klo > ilo:
            ilo = klo
        if khi < ihi:
            ihi = khi
    return RichResult(payload={
        "lower": tot_lo, "upper": tot_hi, "width": tot_hi - tot_lo,
        "inter_lower": ilo, "inter_upper": ihi, "n_strata": len(levels), "n": n,
        "method": "Manski no-assumption ATE bounds within covariate strata"})


manskinoassumptionoutcome = manski_no_assumption_outcome


def cheatsheet():
    return "mskbnd2: Manski no-assumption ATE bounds within covariate strata"
