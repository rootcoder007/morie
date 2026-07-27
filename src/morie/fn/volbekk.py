# morie.fn -- function file (rootcoder007/morie)
"""BEKK multivariate volatility (panel front-end)."""

from ._garch import bekk_fit
from ._richresult import RichResult

__all__ = ["vol_bekk_garch"]


def vol_bekk_garch(R_panel):
    r"""Scalar BEKK(1,1) multivariate GARCH.

    .. math:: H_t = C'C + A'\epsilon_{t-1}\epsilon_{t-1}'A
              + B'H_{t-1}B

    The quadratic form makes every :math:`H_t` positive definite by
    construction, which is what BEKK buys over a naive
    element-by-element multivariate GARCH. Fitted with variance
    targeting: :math:`C'C = \bar H(1 - a - b)`, so only the two
    dynamic parameters are estimated and the long-run covariance
    equals the sample covariance exactly.

    Parameters
    ----------
    R_panel : array-like, shape (T, k)
        Return panel, k >= 2 series.

    Returns
    -------
    RichResult
        keys: ``H`` (T, k, k conditional covariances), ``a``, ``b``,
        ``persistence``, ``H_bar``, ``C``, ``loglik``, ``T``, ``k``,
        ``converged``, ``method``.

    References
    ----------
    Engle, R. F. & Kroner, K. F. (1995). Multivariate simultaneous
    generalized ARCH. *Econometric Theory*, 11(1), 122-150.

    Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd ed.).
    Wiley. Ch. 10 (multivariate volatility models).
    """
    return RichResult(payload=bekk_fit(R_panel))


def cheatsheet():
    return "volbekk: scalar BEKK(1,1), H_t PD by construction, variance targeting"
