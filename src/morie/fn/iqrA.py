# morie.fn -- function file (rootcoder007/morie)
"""Tukey fences on the hinges."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["iqr_outlier"]


def _hinges(x):
    """Tukey lower and upper fourths, the fivenum hinges."""
    x = sorted(x)
    n = len(x)
    n4 = math_floor((n + 3) / 2.0) / 2.0
    def at(d):
        lo = int(math_floor(d)) - 1
        hi = int(math_ceil(d)) - 1
        return 0.5 * (x[lo] + x[hi])
    return at(n4), at(n + 1 - n4)


def math_floor(v):
    return float(int(v) if v >= 0 or v == int(v) else int(v) - 1)


def math_ceil(v):
    return float(int(v) if v <= 0 or v == int(v) else int(v) + 1)


def iqr_outlier(x, k=1.5):
    """Flag points outside the Tukey fences.

    Tukey rule is deliberately not a test: there is no null, no p-value
    and no distributional assumption, only a pair of fences drawn from
    the middle half of the data.  That is why it survives on skewed
    data where a three-sigma rule quietly flags the entire tail.  The
    hinges are used rather than sample quantiles, because the hinges
    are what Tukey defined and what a boxplot draws.

    Formula: flag ``x < H_L - k (H_U - H_L)`` or
    ``x > H_U + k (H_U - H_L)``, with ``k = 1.5`` for outliers and
    ``k = 3`` for the far-out points.

    Parameters
    ----------
    x : array-like
        Sample.
    k : float, default 1.5
        Fence multiplier.

    Returns
    -------
    RichResult
        ``estimate`` (proportion flagged), ``n_out``, ``lower``,
        ``upper``, ``iqr``, ``flags``, ``n``.

    References
    ----------
    Tukey, J. W. (1977).  Exploratory Data Analysis.  Addison-Wesley,
    chapter 2 (the hinges) and chapter 3 (the fences).
    """
    v = C.vec(x)
    n = len(v)
    hl, hu = _hinges(v)
    spread = hu - hl
    lo = hl - k * spread
    hi = hu + k * spread
    flags = [1.0 if (t < lo or t > hi) else 0.0 for t in v]
    nout = int(sum(flags))
    return RichResult(payload={
        "estimate": nout / n, "n_out": nout, "lower": lo, "upper": hi,
        "iqr": spread, "flags": flags, "n": n,
        "method": "Tukey fences on the hinges"})


iqroutlier = iqr_outlier


def cheatsheet():
    return "iqrA: Tukey fences on the hinges."
