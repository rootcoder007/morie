# morie.fn -- function file (rootcoder007/morie)
"""Population total from a simple random sample."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["srstotal", "survey_total"]


def srstotal(y, N, level=0.95):
    """Estimate a population total from a simple random sample.

    The total is the mean scaled by N, so its variance is scaled by
    N^2 -- which is why a total looks so much less precise than the
    mean it came from even though they carry identical information.
    The coefficient of variation, which is identical for the two, is
    returned to make that visible.

    Formula: Yhat = N ybar;  v(Yhat) = N^2 (1 - f) s^2 / n,  f = n/N

    Parameters
    ----------
    y : array-like
        Sample observations.
    N : float
        Population size.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``, ``mean``,
        ``cv``, ``fpc``, ``N``, ``n``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 2:
    Yhat = N ybar with V(Yhat) = N^2 (N - n)/N S^2/n for simple random
    sampling without replacement.  Cross-checked against the reference
    implementation in the CRAN package ``samplingbook`` 1.2.4, whose
    ``Smean`` uses the same finite-population-corrected variance for
    the mean.
    """
    y = C.vec(y)
    n = len(y)
    if n < 2:
        raise ValueError("a variance needs at least two observations")
    N = float(N)
    if N < n:
        raise ValueError("N must be at least n")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    m = sum(y) / n
    s2 = C.var(y, 1)
    k = 1.0 if math.isinf(N) else (N - n) / N
    var = N * N * k * s2 / n
    se = math.sqrt(var)
    est = N * m
    z = C.qnorm((1.0 + float(level)) / 2.0)
    return RichResult(payload={
        "estimate": est, "se": se, "ci_lower": est - z * se,
        "ci_upper": est + z * se, "mean": m,
        "cv": se / est if est != 0 else float("nan"), "fpc": k,
        "N": N, "n": n,
        "method": "SRS population total, Yhat = N ybar"})


survey_total = srstotal


def cheatsheet():
    return "smltot: Yhat = N ybar; v = N^2 (1-f) s^2/n"
