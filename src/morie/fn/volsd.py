# morie.fn -- function file (rootcoder007/morie)
"""Rolling-window volatility from squared returns."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_simple_diff"]


def vol_simple_diff(r, window=20):
    r"""Rolling root-mean-square volatility.

    .. math:: \hat\sigma_t = \sqrt{\tfrac1w \sum_{s=t-w+1}^{t} r_s^2},

    the zero-mean historical estimator every volatility comparison
    starts from. The first ``window - 1`` entries are NaN rather than
    silently shorter.

    Parameters
    ----------
    r : array-like, shape (n,)
        Return series.
    window : int, default 20

    Returns
    -------
    RichResult
        keys: ``sigma`` (n, NaN-padded), ``sigma2``, ``window``,
        ``n``, ``method``.

    References
    ----------
    Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd
    ed.). Wiley. Ch. 3 (historical volatility as the baseline).
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    w = int(window)
    if w < 2:
        raise ValueError(f"window must be at least 2, got {w}.")
    if n < w:
        raise ValueError(f"need at least window = {w} returns, got {n}.")

    c = np.concatenate([[0.0], np.cumsum(r**2)])
    s2 = np.full(n, np.nan)
    s2[w - 1 :] = (c[w:] - c[:-w]) / w

    return RichResult(
        payload={
            "sigma": np.sqrt(s2),
            "sigma2": s2,
            "window": w,
            "n": int(n),
            "method": f"Rolling RMS volatility (window = {w})",
        }
    )


def cheatsheet():
    return "volsd: sigma_t = sqrt(mean r^2 over the trailing window)"
