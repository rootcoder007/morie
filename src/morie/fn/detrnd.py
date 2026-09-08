# morie.fn -- function file (rootcoder007/morie)
"""Linear detrending of a series.

Standard preprocessing.  Triage confirmed this names no owning
source; ordinary least squares on time is implemented and no citation
is manufactured.
"""

from ._richresult import RichResult, with_describe_pointer

__all__ = ["detrend_climate"]


def detrend_climate(x, t=None):
    """Remove a straight line fitted by least squares,

        residual_i = x_i - (a + b t_i).

    The slope is the usual covariance over variance, so the result is
    the series with its linear trend taken out and its mean removed.
    ``t`` defaults to 0, 1, 2, ..., which is the right choice only for
    evenly spaced observations; pass the real times when they are not.

    Parameters
    ----------
    x : array-like series.
    t : optional array-like times; 0..n-1 by default.

    Returns
    -------
    RichResult with keys estimate (the fitted slope), detrended,
    fitted, intercept, slope, n, method.
    """
    v = [float(u) for u in x]
    n = len(v)
    if n < 2:
        raise ValueError("need at least two observations")
    tv = [float(u) for u in t] if t is not None else [float(i) for i in range(n)]
    if len(tv) != n:
        raise ValueError("t and x must have the same length")
    tb = sum(tv) / n
    xb = sum(v) / n
    stt = sum((u - tb) ** 2 for u in tv)
    if stt == 0:
        raise ValueError("t must not be constant")
    b = sum((tv[i] - tb) * (v[i] - xb) for i in range(n)) / stt
    a = xb - b * tb
    fit = [a + b * u for u in tv]
    return with_describe_pointer(RichResult(payload={
        "estimate": float(b),
        "detrended": [v[i] - fit[i] for i in range(n)],
        "fitted": fit, "intercept": float(a), "slope": float(b), "n": n,
        "method": "linear detrending by least squares",
    }), "detrnd")


def cheatsheet():
    return "detrnd: Linear detrending"


# compact alias per ledger/NAMING.md
detrendlin = detrend_climate
