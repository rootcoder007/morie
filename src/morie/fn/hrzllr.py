# morie.fn -- function file (rootcoder007/morie)
"""Local linear regression."""

import numpy as np

from ._horowitz import local_linear
from ._richresult import RichResult

__all__ = ["hrz_local_linear"]


def hrz_local_linear(x, y, grid=None, h=None, kernel_name="gaussian"):
    r"""Local linear regression (Horowitz Ch. 2):

    .. math:: (\hat\alpha, \hat\beta) = \arg\min
              \sum_i K_h(x - X_i)\big(Y_i - \alpha
              - \beta(X_i - x)\big)^2,

    with :math:`\hat m(x) = \hat\alpha`. Fitting a local LINE
    rather than a local constant makes the bias O(h^2) uniformly,
    including at the boundary -- the automatic boundary correction
    that motivates preferring it over Nadaraya-Watson. The local slope
    is returned too, since it estimates m'(x) for free.

    Parameters
    ----------
    x, y : array-like
        Regressor and response.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``grid``, ``fitted``, ``slope``, ``bandwidth``, ``n``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (local polynomial regression).
    """
    g, m, b, hh = local_linear(x, y, grid=grid, h=h, name=kernel_name)
    return RichResult(payload={"grid": g, "fitted": m, "slope": b,
                               "bandwidth": hh, "n": int(np.asarray(x).size),
                               "method": "Local linear; O(h^2) bias including at the boundary"})


def cheatsheet():
    return "hrzllr: boundary bias fixed automatically; slope estimates m'(x) free"
