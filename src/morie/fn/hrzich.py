# morie.fn -- function file (rootcoder007/morie)
"""Ichimura estimator (front-end)."""

import numpy as np

from ._horowitz import silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_ichimura"]


from .hrznls import hrz_semiparametric_ls


def hrz_ichimura(X, y, h=None, kernel_name="gaussian"):
    r"""Ichimura's semiparametric least-squares estimator (Horowitz
    Ch. 2), delegating to :mod:`morie.fn.hrznls`:

    .. math:: \hat\beta = \arg\min_{b:\,|b_1|=1}
              \sum_i \big(Y_i - \hat G_{-i,b}(X_i'b)\big)^2.

    Named entry point for the same criterion. The estimator is root-n
    consistent and asymptotically normal even though the link G is
    estimated nonparametrically, which is what distinguishes a
    semiparametric problem from a purely nonparametric one.

    Parameters
    ----------
    X, y : array-like
        Covariates and response.
    h : float, optional
        Bandwidth.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``beta``, ``sse``, ``converged``, ``root_n`` (True),
        ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (Ichimura 1993).
    """
    out = hrz_semiparametric_ls(X, y, h=h, kernel_name=kernel_name)
    return RichResult(payload={"beta": out["beta"], "sse": out["sse"],
                               "converged": out["converged"], "root_n": True,
                               "n": out["n"], "d": out["d"],
                               "method": "Ichimura SLS; beta root-n despite nonparametric G"})


def cheatsheet():
    return "hrzich: beta is root-n even though G is not -- the semiparametric point"
