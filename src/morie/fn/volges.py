# morie.fn -- function file (rootcoder007/morie)
"""GARCH-implied expected shortfall."""

from ._garch import var_es
from ._richresult import RichResult

__all__ = ["vol_garch_es_impl"]


def vol_garch_es_impl(mu, sigma_next, alpha=0.05, dist="normal", nu=8.0):
    r"""One-step expected shortfall from a volatility forecast.

    .. math:: ES_\alpha = -\mu + \sigma \phi(z_\alpha)/\alpha

    for the Gaussian case: the mean loss *given* the loss already
    exceeds VaR. Unlike VaR, ES is subadditive, so it does not
    penalise diversification -- the reason regulators moved to it.
    ES always exceeds the VaR at the same alpha, which the tests pin.

    Parameters
    ----------
    mu : float
        Conditional mean forecast.
    sigma_next : float
        One-step volatility.
    alpha : float, default 0.05
        Tail probability.
    dist : {"normal", "t"}
        Innovation distribution.
    nu : float, default 8.0
        Degrees of freedom for the t.

    Returns
    -------
    RichResult
        keys: ``es``, ``var``, ``quantile``, ``alpha``, ``dist``.

    References
    ----------
    Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). Coherent
    measures of risk. *Mathematical Finance*, 9(3), 203-228.

    Tsay, R. S. (2010). *Analysis of Financial Time Series* (3rd ed.).
    Wiley. Ch. 7.
    """
    return RichResult(payload=var_es(mu, sigma_next, alpha, dist, nu))


def cheatsheet():
    return "volges: ES = mean loss beyond VaR; subadditive, always >= VaR"
