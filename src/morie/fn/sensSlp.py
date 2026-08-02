# morie.fn -- function file (rootcoder007/morie)
"""Sen's slope for a time-indexed series."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sens_slope", "sen_slope"]


def sens_slope(y, t=None, alpha=0.05):
    """Sen's slope: the Theil-Sen estimator applied to a time series
    against its (possibly implicit) time index, the form in which
    hydrology and climatology cite Sen (1968), usually alongside the
    Mann-Kendall trend test.

    One estimator, one implementation: the computation is
    :func:`morie.fn.theils.theil_sen` on ``(t, y)``, so the two
    entries cannot drift apart. Everything said there -- the excluded
    tied pairs, the 29.3% breakdown, the order-statistic confidence
    interval that needs no residual variance -- applies verbatim.
    The one addition is the trend reading: with a regular time index
    the slope is per time step, and ``trend`` states its sign only
    when the Sen interval excludes zero.

    Parameters
    ----------
    y : array-like, shape (n,)
        Series values.
    t : array-like, optional
        Time index; ``0..n-1`` when omitted.
    alpha : float, default 0.05
        Miss probability for the interval.

    Returns
    -------
    RichResult
        keys: everything from :func:`morie.fn.theils.theil_sen`, plus
        ``trend`` ("increasing", "decreasing" or "no trend at this "
        "alpha") and ``per``.

    References
    ----------
    Sen, P. K. (1968), *JASA* 63:1379-1389. Theil, H. (1950),
    *Proc. KNAW* 53. Mann, H. B. (1945), *Econometrica* 13:245-259,
    for the companion trend test.
    """
    from .theils import theil_sen

    yv = np.asarray(y, dtype=float).ravel()
    tv = np.arange(yv.size, dtype=float) if t is None else \
        np.asarray(t, dtype=float).ravel()
    out = theil_sen(tv, yv, alpha=alpha)
    lo, hi = out["ci"]
    trend = "increasing" if lo > 0 else (
        "decreasing" if hi < 0 else "no trend at this alpha")
    payload = dict(out)
    payload["trend"] = trend
    payload["per"] = "time step" if t is None else "unit of t"
    payload["alias_of"] = "morie.fn.theils.theil_sen"
    payload["method"] = "Sen's slope: Theil-Sen against the time index"
    return RichResult(payload=payload)


def cheatsheet():
    return "sensSlp: Theil-Sen on (t, y) -- one implementation, trend read off Sen's interval"


#: Catalogue alias for :func:`sens_slope`.
sen_slope = sens_slope
