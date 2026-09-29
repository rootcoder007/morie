# morie.fn -- function file (rootcoder007/morie)
"""MGWR local residuals."""

from .mgwrfit import mgwrfit


def mgwrres(y, X, coords, bws=None, kernel="bisquare", adaptive=False):
    r"""MGWR residuals ``y_i - sum_k beta_k(u_i) x_ik`` at convergence of the backfitting.

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
    >>> round(mgwrres(y, X, P, bws=[6.0, 3.0, 8.0], kernel="gaussian")[4], 10)
    -0.1859569274
    """
    return mgwrfit(y, X, coords, bandwidths=bws, kernel=kernel, adaptive=adaptive)["residuals"]


mgwrres_fn = mgwrres


def cheatsheet() -> str:
    return "mgwrres(y, X, coords, bws) -> MGWR residuals."
