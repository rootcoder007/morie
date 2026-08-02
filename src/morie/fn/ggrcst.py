# morie.fn -- function file (rootcoder007/morie)
"""Granger causality test."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["granger_causality"]


def _lag_design(y, x, p, include_x):
    n = y.size
    cols = [np.ones(n - p)]
    for j in range(1, p + 1):
        cols.append(y[p - j : n - j])
    if include_x:
        for j in range(1, p + 1):
            cols.append(x[p - j : n - j])
    return np.column_stack(cols), y[p:]


def _rss(D, t):
    b, *_ = np.linalg.lstsq(D, t, rcond=None)
    r = t - D @ b
    return float((r**2).sum())


def granger_causality(x, y, p=1):
    r"""Does x Granger-cause y? F-test of restricted vs unrestricted AR.

    Restricted model: :math:`y_t` on p own lags. Unrestricted: plus p
    lags of x. Under the null that the x-lag coefficients are all
    zero,

    .. math:: F = \frac{(\mathrm{RSS}_r - \mathrm{RSS}_u)/p}
              {\mathrm{RSS}_u/(m - 2p - 1)} \sim F_{p,\, m-2p-1},

    with m = n - p usable observations. Rejection means x's past
    improves the prediction of y beyond y's own past -- Granger's
    operational notion of causality (predictive, not structural: a
    common driver of both series produces it too).

    Parameters
    ----------
    x, y : array-like, shape (n,)
        The candidate cause and the response series.
    p : int, default 1
        Lag order.

    Returns
    -------
    RichResult
        keys: ``statistic`` (F), ``p_value``, ``df`` (p, m - 2p - 1),
        ``rss_restricted``, ``rss_unrestricted``, ``n``, ``p_lags``,
        ``method``.

    References
    ----------
    Granger, C. W. J. (1969). Investigating causal relations by
    econometric models and cross-spectral methods. *Econometrica*,
    37(3), 424-438.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have equal length.")
    p = int(p)
    if p < 1:
        raise ValueError(f"p must be at least 1, got {p}.")
    n = y.size
    m = n - p
    dof2 = m - 2 * p - 1
    if dof2 < 1:
        raise ValueError(f"need at least {3 * p + 2} observations for p = {p}, got {n}.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite.")

    Dr, t = _lag_design(y, x, p, include_x=False)
    Du, _ = _lag_design(y, x, p, include_x=True)
    rss_r, rss_u = _rss(Dr, t), _rss(Du, t)
    if rss_u <= 0:
        raise ValueError("unrestricted model fits exactly; F statistic undefined.")
    F = ((rss_r - rss_u) / p) / (rss_u / dof2)
    pv = float(stats.f.sf(F, p, dof2))

    return RichResult(
        payload={
            "statistic": float(F),
            "p_value": pv,
            "df": (p, dof2),
            "rss_restricted": rss_r,
            "rss_unrestricted": rss_u,
            "n": int(n),
            "p_lags": p,
            "method": f"Granger causality F-test (p={p})",
        }
    )


def cheatsheet():
    return "ggrcst: F-test of x-lags in y's AR (Granger 1969)"
