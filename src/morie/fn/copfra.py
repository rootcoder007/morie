# morie.fn -- function file (rootcoder007/morie)
"""Frank copula (no tail dependence)."""

import numpy as np

from ._copula import copula_cdf, copula_tau
from ._richresult import RichResult

__all__ = ["frank_copula"]


def frank_copula(u, v, theta):
    r"""Frank copula (no tail dependence) CDF and its Kendall's tau.

    Evaluates :math:`C(u, v)` for the frank family via the shared
    core in :mod:`morie.fn._copula`, together with the Kendall's tau
    implied by the parameter (Czado 2019, Table 3.2, p. 54 -- read in
    the library PDF). Parameter range: ``theta != 0``.

    Parameters
    ----------
    u, v : array-like in [0, 1]
        Uniform margins (broadcastable).
    theta : float
        Copula parameter.

    Returns
    -------
    RichResult
        keys: ``cdf`` (same shape as the broadcast u, v), ``tau``,
        ``theta``, ``family``, ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 3 (bivariate copula classes), Table 3.2 p. 54
    (parameter/Kendall's tau relations).
    """
    cdf = copula_cdf("frank", u, v, theta)
    return RichResult(
        payload={
            "cdf": cdf,
            "tau": copula_tau("frank", theta),
            "theta": float(theta),
            "family": "frank",
            "method": "Frank copula (no tail dependence) CDF (Czado 2019 Ch. 3)",
        }
    )


def cheatsheet():
    return "copfra: frank copula CDF + Kendall tau (theta != 0)"
