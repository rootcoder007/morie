# morie.fn -- function file (rootcoder007/morie)
"""Nadaraya-Watson regression."""

import numpy as np

from ._horowitz import nw_regression
from ._richresult import RichResult

__all__ = ["hrz_nw_regression", "horowitz_nw_regression"]


def hrz_nw_regression(x, y, grid=None, h=None, kernel_name="gaussian"):
    r"""Nadaraya-Watson kernel regression (Horowitz Ch. 2):

    .. math:: \hat m(x) = \frac{\sum_i K_h(x - X_i) Y_i}
                                {\sum_i K_h(x - X_i)}.

    A local CONSTANT fit. Its bias is O(h^2) in the interior but only
    O(h) at the boundary, where the kernel window becomes one-sided --
    exactly the defect local linear regression
    (:mod:`morie.fn.hrzllr`) was designed to remove.

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
        keys: ``grid``, ``fitted``, ``bandwidth``, ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (kernel regression).
    """
    g, m, hh = nw_regression(x, y, grid=grid, h=h, name=kernel_name)
    return RichResult(payload={"grid": g, "fitted": m, "bandwidth": hh,
                               "n": int(np.asarray(x).size),
                               "method": "NW local constant; O(h) boundary bias"})


def cheatsheet():
    return "hrznwr: local constant, so the boundary bias is O(h) not O(h^2)"


#: Catalogue alias for :func:`hrz_nw_regression`.
horowitz_nw_regression = hrz_nw_regression
