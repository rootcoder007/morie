# morie.fn -- function file (rootcoder007/morie)
"""MGWR per-variable bandwidth selection."""

from .mgwrfit import _backfit, _prep


def mgwrbw(y, X, coords, kernel="bisquare", adaptive=False, threshold=1e-8, max_iter=200):
    r"""Covariate-specific MGWR bandwidths selected inside the backfitting (Fotheringham, Yang and Kang 2017).

    At every backfitting pass the bandwidth of covariate k is chosen by
    golden-section search over [min d_ij, max d_ij] minimising the AICc
    of the one-covariate GWR (no intercept) of its current partial residual,
    n log(RSS/n) + n log(2 pi) + n (n + tr S_k) / (n - 2 - tr S_k); the
    passes stop when both the relative change of the RSS and of every
    bandwidth fall below threshold. Returns the bandwidths (intercept
    first).

    References
    ----------
    Fotheringham, A. S., Yang, W. and Kang, W. (2017). Multiscale
    geographically weighted regression (MGWR). *Annals of the American
    Association of Geographers* 107, 1247-1265.
    Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W. and Wolf, L. J.
    (2020). Inference in multiscale geographically weighted regression.
    *Geographical Analysis* 52, 87-106.

    Examples
    --------
    >>> import math
    >>> P = [(float(i % 5), float(i // 5)) for i in range(20)]
    >>> X = [[math.sin(i), (0.3 * i) % 1.1] for i in range(20)]
    >>> y = [1.0 + (1 + 0.2 * P[i][0]) * X[i][0] - X[i][1] + 0.1 * math.cos(3 * i) for i in range(20)]
    >>> [round(b, 4) for b in mgwrbw(y, X, P, kernel="gaussian")]
    [5.0, 1.0017, 5.0]
    """
    yv, Xm, D = _prep(y, X, coords)
    return _backfit(yv, Xm, D, None, kernel, adaptive, threshold, max_iter, True, False)[4]


mgwrbw_fn = mgwrbw


def cheatsheet() -> str:
    return "mgwrbw(y, X, coords) -> per-covariate MGWR bandwidths by golden-section AICc inside backfitting."
