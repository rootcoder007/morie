# morie.fn -- function file (rootcoder007/morie)
"""Weighted variance and weighted mean of a sample carrying unit weights."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["weighted_variance"]


def weighted_variance(y, weights=None):
    """Weighted sample mean and weighted sample variance.

    The weights are *reliability* weights: an observation with weight
    ``w`` counts as ``w`` observations, so the denominator is
    ``(sum w) - 1`` rather than ``n - 1``.  With every weight equal to
    one this collapses exactly onto the unbiased sample variance, which
    is the anchor used in the tests.

    Formula: ``ybar_w = sum(w_i y_i) / sum(w_i)`` and
    ``s2_w = sum(w_i (y_i - ybar_w)^2) / (sum(w_i) - 1)``.

    Parameters
    ----------
    y : array-like
        Observations.
    weights : array-like, optional
        Non-negative weights; equal weights if omitted.

    Returns
    -------
    RichResult
        ``estimate`` (weighted variance), ``mean``, ``sd``, ``se`` of the
        weighted mean, ``sumw``, ``n``, ``method``.

    References
    ----------
    Lohr, S. L. (2010).  Sampling: Design and Analysis, 2nd edition.
    Brooks/Cole, section 7.2 (weighted estimation).
    """
    yy = C.vec(y)
    n = len(yy)
    if n == 0:
        raise ValueError("weighted_variance: y is empty")
    if weights is None:
        w = [1.0] * n
    else:
        w = C.vec(weights)
    if len(w) != n:
        raise ValueError("weighted_variance: y and weights differ in length")
    for v in w:
        if v < 0.0:
            raise ValueError("weighted_variance: weights must be non-negative")
    sw = sum(w)
    if sw <= 0.0:
        raise ValueError("weighted_variance: weights sum to zero")
    mu = sum(w[i] * yy[i] for i in range(n)) / sw
    ss = sum(w[i] * (yy[i] - mu) ** 2 for i in range(n))
    s2 = ss / (sw - 1.0) if sw > 1.0 else float("nan")
    sd = math.sqrt(s2) if s2 == s2 and s2 >= 0.0 else float("nan")
    # design-free standard error of the weighted mean
    sw2 = sum(v * v for v in w)
    se = math.sqrt(s2 * sw2 / (sw * sw)) if s2 == s2 else float("nan")
    return RichResult(payload={
        "estimate": float(s2), "mean": float(mu), "sd": float(sd),
        "se": float(se), "sumw": float(sw), "n": n,
        "method": "weighted variance, sum(w (y-ybar_w)^2)/(sum(w)-1) [Lohr 2010]"})


# CANONICAL TEST
# >>> r = weighted_variance([1.0, 2.0, 3.0, 10.0], None)
# >>> assert abs(r["estimate"] - 16.0) < 1e-12   # == var(c(1,2,3,10))
# >>> assert abs(r["mean"] - 4.0) < 1e-12


def cheatsheet():
    return "wvar(y, weights): weighted mean and variance, sum(w)-1 denominator."
