# morie.fn -- function file (rootcoder007/morie)
"""MGWR local coefficient estimates."""

from .mgwrfit import mgwrfit


def mgwrcof(y, X, coords, bws=None, kernel="bisquare", adaptive=False):
    r"""Local MGWR coefficients ``beta_k(u_i)``, one row per location (intercept first).

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
    >>> round(mgwrcof(y, X, P, bws=[6.0, 3.0, 8.0], kernel="gaussian")[0][1], 10)
    1.3394448553
    """
    return mgwrfit(y, X, coords, bandwidths=bws, kernel=kernel, adaptive=adaptive)["betas"]


mgwrcof_fn = mgwrcof


def cheatsheet() -> str:
    return "mgwrcof(y, X, coords, bws) -> MGWR local coefficients."
