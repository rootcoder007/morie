# morie.fn -- function file (rootcoder007/morie)
"""Two-scale realised variance (TSRV)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_two_scale_rv"]


def vol_two_scale_rv(r_intraday, K=5):
    r"""Zhang-Mykland-Ait-Sahalia two-scale estimator.

    .. math:: TSRV = RV^{(avg,K)} - \frac{\bar n}{n} RV^{(all)},
              \qquad \bar n = \frac{n - K + 1}{K},

    the average of K subsampled slow-scale RVs, bias-corrected by the
    fast-scale RV: under i.i.d. microstructure noise the fast RV
    explodes with the noise variance, and subtracting the right
    multiple of it recovers a consistent estimate of integrated
    variance -- the first consistent noise-robust RV.

    Parameters
    ----------
    r_intraday : array-like, shape (n,)
        Intraday returns at the finest grid.
    K : int, default 5
        Number of subsampling offsets (slow scale = every K-th tick).

    Returns
    -------
    RichResult
        keys: ``tsrv``, ``rv_fast`` (all-data RV), ``rv_slow_avg``,
        ``K``, ``n_returns``, ``method``.

    References
    ----------
    Zhang, L., Mykland, P. A. & Ait-Sahalia, Y. (2005). A tale of two
    time scales: determining integrated volatility with noisy
    high-frequency data. *JASA*, 100(472), 1394-1411.
    """
    r = np.asarray(r_intraday, dtype=float).ravel()
    n = r.size
    K = int(K)
    if K < 2:
        raise ValueError(f"K must be at least 2, got {K}.")
    if n < 2 * K:
        raise ValueError(f"need at least 2K = {2 * K} returns, got {n}.")

    # prices implied by cumulating returns; subsample at each offset
    p = np.concatenate([[0.0], np.cumsum(r)])
    rv_fast = float((r**2).sum())
    slow = []
    for off in range(K):
        sub = p[off::K]
        if sub.size >= 2:
            slow.append(float((np.diff(sub) ** 2).sum()))
    rv_slow = float(np.mean(slow))
    nbar = (n - K + 1) / K
    tsrv = rv_slow - (nbar / n) * rv_fast

    return RichResult(
        payload={
            "tsrv": tsrv,
            "rv_fast": rv_fast,
            "rv_slow_avg": rv_slow,
            "K": K,
            "n_returns": int(n),
            "method": f"Two-scale realised variance (K = {K})",
        }
    )


def cheatsheet():
    return "voltsr: avg subsampled RV - (nbar/n) RV_all (ZMA 2005)"
