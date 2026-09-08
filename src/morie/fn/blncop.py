# morie.fn -- function file (rootcoder007/morie)
"""Blomqvist's beta (medial correlation) from a copula."""

from ._copula import FAMILIES, copula_cdf
from ._richresult import RichResult

__all__ = ["blomqvists_beta_copula"]


def blomqvists_beta_copula(family, theta=None, nu=None):
    r"""Blomqvist's beta.

    .. math:: \beta = 4\,C(\tfrac12, \tfrac12) - 1,

    the medial correlation: it depends on the copula *only at the
    centre*, which makes it trivially computable and completely blind
    to tail behaviour -- the trade-off worth stating next to tau and
    rho, which integrate over the whole square. beta = 0 for
    independence and 1 for the comonotone copula.

    Parameters
    ----------
    family : str
        Copula family.
    theta : float
        Parameter.
    nu : float, optional
        t degrees of freedom.

    Returns
    -------
    RichResult
        keys: ``beta``, ``c_half``, ``family``, ``theta``,
        ``method``.

    References
    ----------
    Blomqvist, N. (1950). On a measure of dependence between two
    random variables. *The Annals of Mathematical Statistics*, 21(4),
    593-600.

    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 2 (dependence measures).
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    c = float(copula_cdf(family, 0.5, 0.5, theta, nu))
    return RichResult(
        payload={
            "beta": 4.0 * c - 1.0,
            "c_half": c,
            "family": family,
            "theta": None if theta is None else float(theta),
            "method": "Blomqvist's beta = 4 C(1/2, 1/2) - 1",
        }
    )


def cheatsheet():
    return "blncop: beta = 4 C(0.5, 0.5) - 1 -- centre only, blind to tails"
