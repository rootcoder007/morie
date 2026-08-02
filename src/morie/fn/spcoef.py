# morie.fn -- function file (rootcoder007/morie)
"""Spearman's rho implied by a copula."""

from . import _array_core as np

from ._copula import FAMILIES, copula_cdf
from ._richresult import RichResult

__all__ = ["spearmans_rho_copula"]


def spearmans_rho_copula(family, theta=None, nu=None, n=200):
    r"""Spearman's rho from a parametric copula.

    .. math:: \rho_S = 12 \int_0^1\!\!\int_0^1 C(u, v)\,du\,dv - 3,

    evaluated on an n x n midpoint grid. For elliptical copulas the
    exact value is :math:`\tfrac6\pi \arcsin(\rho/2)`, which the
    Gaussian branch returns in closed form instead of quadrature; the
    Archimedean families have no such simplification in Czado's
    treatment, so they use the integral -- and the returned
    ``exact`` flag says which route was taken.

    Parameters
    ----------
    family : str
        Copula family.
    theta : float
        Parameter (rho for gaussian/t).
    nu : float, optional
        t degrees of freedom.
    n : int, default 200
        Grid resolution for the numeric route.

    Returns
    -------
    RichResult
        keys: ``rho_s``, ``exact`` (bool), ``family``, ``theta``,
        ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Sec. 2.2 (Kendall's tau and Spearman's rho), Fig. 2.1
    (the elliptical tau/rho relationship).
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    if family == "independence":
        return RichResult(
            payload={
                "rho_s": 0.0,
                "exact": True,
                "family": family,
                "theta": None,
                "method": "Spearman's rho (independence copula)",
            }
        )
    if family == "gaussian":
        r = float(theta)
        if not -1 < r < 1:
            raise ValueError("rho must lie in (-1, 1).")
        return RichResult(
            payload={
                "rho_s": float(6.0 / np.pi * np.arcsin(r / 2.0)),
                "exact": True,
                "family": family,
                "theta": r,
                "method": "Spearman's rho, elliptical closed form (6/pi) arcsin(rho/2)",
            }
        )
    n = int(n)
    if n < 20:
        raise ValueError(f"n must be at least 20, got {n}.")
    g = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(g, g, indexing="ij")
    C = copula_cdf(family, U, V, theta, nu)
    rho_s = 12.0 * float(np.mean(C)) - 3.0
    return RichResult(
        payload={
            "rho_s": rho_s,
            "exact": False,
            "family": family,
            "theta": None if theta is None else float(theta),
            "method": f"Spearman's rho by grid quadrature (n = {n})",
        }
    )


def cheatsheet():
    return "spcoef: rho_S = 12 int int C du dv - 3; gaussian uses (6/pi) arcsin(rho/2)"
