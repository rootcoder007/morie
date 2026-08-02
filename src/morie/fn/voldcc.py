# morie.fn -- function file (rootcoder007/morie)
"""Dynamic Conditional Correlation MGARCH."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .dccmd import dcc_multivariate_garch

__all__ = ["vol_dcc_garch"]


def vol_dcc_garch(R_panel, init=None):
    r"""Engle (2002) Dynamic Conditional Correlation MGARCH.

    Front-end over :func:`morie.fn.dccmd.dcc_multivariate_garch`, which
    holds the estimator. This module exists for the volatility-family
    naming scheme (``vol*``); it does not carry a second copy of the
    recursion.

    The proxy process and its rescaling are equations (9) and (10) of the
    rmgarch model reference:

    .. math::

        Q_t &= (1 - a - b)\bar Q + a\,z_{t-1}z_{t-1}' + b\,Q_{t-1} \\
        R_t &= \mathrm{diag}(Q_t)^{-1/2}\,Q_t\,\mathrm{diag}(Q_t)^{-1/2}

    with :math:`a, b \ge 0` and :math:`a + b < 1` for stationarity and
    positive definiteness. CCC is the special case :math:`a = b = 0`
    (see :func:`morie.fn.volccc.vol_ccc_garch`).

    Parameters
    ----------
    R_panel : array-like, shape (n, k)
        Return panel, n observations x k assets. k >= 2 is required.
    init : array-like of length 2, optional
        Starting values ``(a, b)`` for the correlation step. Ignored by
        the current estimator, which starts from ``(0.02, 0.95)``;
        accepted so callers written against the ``vol*`` signature do not
        break.

    Returns
    -------
    RichResult
        keys: ``a``, ``b``, ``Q_bar``, ``ll``, plus everything
        :func:`~morie.fn.dccmd.dcc_multivariate_garch` returns.

    References
    ----------
    Engle, R. F. (2002). Dynamic Conditional Correlation: a simple class
    of multivariate generalized autoregressive conditional
    heteroskedasticity models. *Journal of Business & Economic
    Statistics*, 20(3), 339-350.
    """
    del init  # estimator uses fixed, well-conditioned starting values
    res = dcc_multivariate_garch(R_panel)
    payload = dict(res.payload)
    # `vol*` callers expect the short names from this module's contract.
    payload["Q_bar"] = payload["unconditional_correlation"]
    payload["ll"] = payload["loglik"]
    payload["sigmas"] = np.sqrt(payload["conditional_variance"])
    return RichResult(title="DCC-GARCH (Engle 2002)", payload=payload)


def cheatsheet():
    return "voldcc: Dynamic Conditional Correlation MGARCH"
