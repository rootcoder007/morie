# morie.fn -- function file (rootcoder007/morie)
"""Gaussian (normal) copula."""

from . import _array_core as np

from ._copula import copula_cdf, copula_tau
from ._richresult import RichResult

__all__ = ["gaussian_copula"]


def gaussian_copula(u, v, rho):
    r"""Gaussian (normal) copula CDF and its Kendall's tau.

    Evaluates :math:`C(u, v)` for the gaussian family via the shared
    core in :mod:`morie.fn._copula`, together with the Kendall's tau
    implied by the parameter (Czado 2019, Table 3.2, p. 54 -- read in
    the library PDF). Parameter range: ``-1 < rho < 1``.

    Parameters
    ----------
    u, v : array-like in [0, 1]
        Uniform margins (broadcastable).
    rho : float
        Copula parameter.

    Returns
    -------
    RichResult
        keys: ``cdf`` (same shape as the broadcast u, v), ``tau``,
        ``rho``, ``family``, ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 3 (bivariate copula classes), Table 3.2 p. 54
    (parameter/Kendall's tau relations).
    """
    cdf = copula_cdf("gaussian", u, v, rho)
    return RichResult(
        payload={
            "cdf": cdf,
            "tau": copula_tau("gaussian", rho),
            "rho": float(rho),
            "family": "gaussian",
            "method": "Gaussian (normal) copula CDF (Czado 2019 Ch. 3)",
        }
    )


def cheatsheet():
    return "copgau: gaussian copula CDF + Kendall tau (-1 < rho < 1)"


# compact alias per ledger/NAMING.md
gaussiancopula = gaussian_copula
