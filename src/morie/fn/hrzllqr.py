# morie.fn -- function file (rootcoder007/morie)
"""Local linear quantile regression."""

import numpy as np

from ._horowitz import local_linear_quantile
from ._richresult import RichResult

__all__ = ["hrz_local_linear_quantile", "horowitz_local_linear_quantile"]


def hrz_local_linear_quantile(x, y, tau=0.5, grid=None, h=None,
                              kernel_name="gaussian"):
    r"""Local linear quantile regression (Horowitz Ch. 3):

    .. math:: (\hat\alpha, \hat\beta) = \arg\min \sum_i
              K_h(x - X_i)\,\rho_\tau\big(Y_i - \alpha
              - \beta(X_i - x)\big),

    with the check loss :math:`\rho_\tau(u) = u(\tau - 1\{u<0\})`.
    Estimating a conditional QUANTILE rather than the mean is robust
    to heavy tails and, unlike the mean, describes how the whole
    conditional distribution moves with x.

    Parameters
    ----------
    x, y : array-like
        Regressor and response.
    tau : float in (0, 1), default 0.5
        Quantile level.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``grid``, ``quantile``, ``tau``, ``bandwidth``, ``n``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 3 (quantile regression).
    """
    g, q, hh = local_linear_quantile(x, y, tau=tau, grid=grid, h=h,
                                     name=kernel_name)
    return RichResult(payload={"grid": g, "quantile": q, "tau": float(tau),
                               "bandwidth": hh, "n": int(np.asarray(x).size),
                               "method": "Local linear check-loss fit; robust to heavy tails"})


def cheatsheet():
    return "hrzllqr: conditional quantile, not mean -- shows the whole distribution move"


#: Catalogue alias for :func:`hrz_local_linear_quantile`.
horowitz_local_linear_quantile = hrz_local_linear_quantile
