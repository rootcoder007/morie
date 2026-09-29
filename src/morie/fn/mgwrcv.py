# morie.fn -- function file (rootcoder007/morie)
"""MGWR cross-validation score."""

from ._qpcore import ssum
from .mgwrfit import mgwrfit


def mgwrcv(y, X, coords, bws=None, kernel="bisquare", adaptive=False):
    r"""Leave-one-out cross-validation score of an MGWR fit, CV = sum_i (e_i / (1 - S_ii))^2.

    For a linear smoother y_hat = S y the leave-one-out residual is
    e_i / (1 - S_ii) (Hastie and Tibshirani 1990, sec. 3.4), with S
    the MGWR hat matrix of :func:`morie.fn.mgwrfit.mgwrfit`.

    References
    ----------
    Hastie, T. J. and Tibshirani, R. J. (1990). *Generalized Additive
    Models*. Chapman and Hall.
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
    >>> round(mgwrcv(y, X, P, bws=[6.0, 3.0, 8.0], kernel="gaussian"), 10)
    0.721102564
    """
    r = mgwrfit(y, X, coords, bandwidths=bws, kernel=kernel, adaptive=adaptive)
    return ssum((e / (1.0 - h)) ** 2 for e, h in zip(r["residuals"], r["hat_diagonal"]))


mgwrcv_fn = mgwrcv


def cheatsheet() -> str:
    return "mgwrcv(y, X, coords, bws) -> sum (e_i / (1 - S_ii))^2 for the MGWR smoother."
