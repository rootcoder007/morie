# morie.fn -- function file (rootcoder007/morie)
"""MGWR local standard errors."""

from .mgwrfit import mgwrfit


def mgwrstd(y, X, coords, bws=None, kernel="bisquare", adaptive=False):
    r"""Standard errors ``sqrt(sigma^2 diag(C_k C_k^T))`` of the local MGWR coefficients, ``beta_k = C_k y`` from the covariate-specific hat matrices ``R_k = diag(x_k) C_k`` and ``sigma^2 = RSS/(n - tr S)`` (Yu et al. 2020).

    ``bws`` are the per-covariate bandwidths (intercept first); None selects
    them by the MGWR backfitting search. Thin front-end to
    :func:`morie.fn.mgwrfit.mgwrfit`.

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
    >>> round(mgwrstd(y, X, P, bws=[6.0, 3.0, 8.0], kernel="gaussian")[0][1], 10)
    0.0571724353
    """
    return mgwrfit(y, X, coords, bandwidths=bws, kernel=kernel, adaptive=adaptive)["se"]


mgwrstd_fn = mgwrstd


def cheatsheet() -> str:
    return "mgwrstd(y, X, coords, bws) -> MGWR local standard errors (Yu et al. 2020)."
