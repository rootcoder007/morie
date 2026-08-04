# morie.fn -- function file (rootcoder007/morie)
"""Plackett copula (constant cross-product ratio)."""

from . import _array_core as np

from ._copula import copula_cdf, copula_tau
from ._richresult import RichResult

__all__ = ["plackett_copula"]


def plackett_copula(u, v, theta):
    r"""Plackett copula (constant cross-product ratio) CDF and its Kendall's tau.

    Evaluates :math:`C(u, v)` for the plackett family via the shared
    core in :mod:`morie.fn._copula`, together with the Kendall's tau
    implied by the parameter (Czado 2019, Table 3.2, p. 54 -- read in
    the library PDF). Parameter range: ``theta > 0``.

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
    cdf = copula_cdf("plackett", u, v, theta)
    return RichResult(
        payload={
            "cdf": cdf,
            "tau": copula_tau("plackett", theta),
            "theta": float(theta),
            "family": "plackett",
            "method": "Plackett copula (constant cross-product ratio) CDF (Czado 2019 Ch. 3)",
        }
    )


def cheatsheet():
    return "plkt: plackett copula CDF + Kendall tau (theta > 0)"


# compact alias per ledger/NAMING.md
plackettcopula = plackett_copula
