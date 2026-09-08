# morie.fn -- function file (rootcoder007/morie)
"""Rogers-Satchell drift-independent OHLC volatility."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_rogers_satchell"]


def _ohlc(o, h, l, c):
    o, h, l, c = (np.asarray(v, dtype=float).ravel() for v in (o, h, l, c))
    n = o.size
    if not (h.size == n and l.size == n and c.size == n):
        raise ValueError("o, h, l, c must have equal length.")
    if np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(c <= 0):
        raise ValueError("prices must be positive.")
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)):
        raise ValueError("need l <= min(o, c) <= max(o, c) <= h per bar.")
    return o, h, l, c


def vol_rogers_satchell(o, h, l, c):
    r"""Rogers-Satchell per-bar variance.

    .. math:: \hat\sigma^2 = \ln\tfrac{H}{C}\ln\tfrac{H}{O}
              + \ln\tfrac{L}{C}\ln\tfrac{L}{O},

    unbiased *whatever the drift* -- the property Parkinson and
    Garman-Klass lack -- because each term pairs a high/low excursion
    with both endpoints.

    Parameters
    ----------
    o, h, l, c : array-like, shape (n,)
        Open, high, low, close per bar.

    Returns
    -------
    RichResult
        keys: ``sigma2`` (n, per bar), ``sigma2_mean``, ``sigma``
        (sqrt of the mean), ``n``, ``method``.

    References
    ----------
    Rogers, L. C. G. & Satchell, S. E. (1991). Estimating variance
    from high, low and closing prices. *The Annals of Applied
    Probability*, 1(4), 504-512.
    """
    o, h, l, c = _ohlc(o, h, l, c)
    s2 = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    m = float(s2.mean())
    return RichResult(
        payload={
            "sigma2": s2,
            "sigma2_mean": m,
            "sigma": float(np.sqrt(max(m, 0.0))),
            "n": int(o.size),
            "method": "Rogers-Satchell drift-independent OHLC variance",
        }
    )


def cheatsheet():
    return "volrs: ln(H/C)ln(H/O) + ln(L/C)ln(L/O), unbiased under drift (RS 1991)"
