# morie.fn -- function file (rootcoder007/morie)
"""Quantile of a complex survey sample by inversion of the weighted CDF."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["survey_quantile"]


def survey_quantile(y, weights=None, quantile=0.5):
    """Design-weighted quantile with a Woodruff confidence interval.

    The estimator is the inverse of the Horvitz-Thompson estimator of the
    distribution function, ``F_w(t) = sum_{y_i <= t} w_i / sum_i w_i``,
    evaluated as the smallest observed value whose weighted CDF reaches
    ``p``.  This is the definition Francisco and Fuller study; with equal
    weights it is exactly the type-1 (inverse-ECDF) sample quantile,
    which is the anchor used in the tests.

    The interval is Woodruff's: the CDF is a mean of indicators, so its
    linearized variance at the point estimate is
    ``V = sum w_i^2 (I(y_i <= q) - F)^2 / (sum w_i)^2``, and the interval
    endpoints are the CDF inverted at ``p -/+ z sqrt(V)``.

    Formula: ``Q_p = inf{t : F_w(t) >= p}``.

    Parameters
    ----------
    y : array-like
        Observations.
    weights : array-like, optional
        Design weights; equal weights if omitted.
    quantile : float
        Probability in (0, 1).

    Returns
    -------
    RichResult
        ``estimate``, ``se`` (of the weighted CDF at the estimate),
        ``lower``, ``upper``, ``p``, ``F`` (attained weighted CDF),
        ``sumw``, ``n``, ``method``.

    References
    ----------
    Francisco, C. A. & Fuller, W. A. (1991).  Quantile estimation with a
    complex survey design.  The Annals of Statistics 19(1):454-469.
    <https://doi.org/10.1214/aos/1176347993>
    Woodruff, R. S. (1952).  Confidence intervals for medians and other
    position measures.  JASA 47(260):635-646.
    <https://doi.org/10.1080/01621459.1952.10483443>
    """
    yy = C.vec(y)
    n = len(yy)
    if n == 0:
        raise ValueError("survey_quantile: y is empty")
    if weights is None:
        w = [1.0] * n
    else:
        w = C.vec(weights)
    if len(w) != n:
        raise ValueError("survey_quantile: y and weights differ in length")
    p = float(quantile)
    if not (0.0 < p < 1.0):
        raise ValueError("survey_quantile: quantile must lie in (0, 1)")
    ordr = sorted(range(n), key=lambda i: yy[i])
    xs = [yy[i] for i in ordr]
    ws = [w[i] for i in ordr]
    tot = sum(ws)
    if tot <= 0.0:
        raise ValueError("survey_quantile: weights sum to zero")
    cum = []
    run = 0.0
    for v in ws:
        run += v
        cum.append(run / tot)
    q = _inv(xs, cum, p)
    F = _cdf(yy, w, tot, q)
    sw2 = sum(v * v for v in w)
    var = sum(w[i] * w[i] * ((1.0 if yy[i] <= q else 0.0) - F) ** 2 for i in range(n)) / (tot * tot)
    se = math.sqrt(var) if var > 0.0 else 0.0
    z = 1.959963984540054
    lo = _inv(xs, cum, min(max(p - z * se, 1e-12), 1.0 - 1e-12))
    hi = _inv(xs, cum, min(max(p + z * se, 1e-12), 1.0 - 1e-12))
    return RichResult(payload={
        "estimate": float(q), "se": float(se), "lower": float(lo),
        "upper": float(hi), "p": p, "F": float(F), "sumw": float(tot),
        "neff": float(tot * tot / sw2), "n": n,
        "method": "weighted CDF inversion with Woodruff interval [Francisco & Fuller 1991]"})


def _inv(xs, cum, p):
    for i in range(len(xs)):
        if cum[i] >= p:
            return xs[i]
    return xs[-1]


def _cdf(yy, w, tot, t):
    return sum(w[i] for i in range(len(yy)) if yy[i] <= t) / tot


# CANONICAL TEST
# >>> r = survey_quantile([4.0, 1.0, 3.0, 2.0, 5.0], None, 0.5)
# >>> assert abs(r["estimate"] - 3.0) < 1e-12   # == quantile(y, .5, type = 1)
# >>> # a doubled weight is the same as a duplicated observation
# >>> a = survey_quantile([1.0, 2.0, 3.0], [1.0, 2.0, 1.0], 0.5)["estimate"]
# >>> b = survey_quantile([1.0, 2.0, 2.0, 3.0], None, 0.5)["estimate"]
# >>> assert abs(a - b) < 1e-12


def cheatsheet():
    return "svyqtl(y, weights, quantile): weighted-CDF quantile, Woodruff interval."
