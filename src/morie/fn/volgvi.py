# morie.fn -- function file (rootcoder007/morie)
"""GARCH-implied Value-at-Risk."""

from ._garch import var_es
from ._richresult import RichResult

__all__ = ["vol_garch_var_impl"]


def vol_garch_var_impl(mu, sigma_next, alpha=0.05, dist="normal", nu=8.0):
    r"""One-step Value-at-Risk from a volatility forecast.

    .. math:: VaR_\alpha = -(\mu + \sigma_{t+1} z_\alpha),

    reported as a positive loss. The t branch standardises to unit
    variance first, so switching ``dist`` changes the tail shape
    without silently rescaling the volatility forecast that was fed in.

    Parameters
    ----------
    mu : float
        Conditional mean forecast.
    sigma_next : float
        One-step volatility (standard deviation, not variance).
    alpha : float, default 0.05
        Tail probability.
    dist : {"normal", "t"}
        Innovation distribution.
    nu : float, default 8.0
        Degrees of freedom for the t.

    Returns
    -------
    RichResult
        keys: ``var``, ``es``, ``quantile``, ``alpha``, ``dist``.

    References
    ----------
    Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd ed.).
    Wiley. Ch. 7 (extreme values, quantile estimation, and
    value at risk).
    """
    return RichResult(payload=var_es(mu, sigma_next, alpha, dist, nu))


def cheatsheet():
    return "volgvi: VaR = -(mu + sigma z_alpha), positive-loss convention"
