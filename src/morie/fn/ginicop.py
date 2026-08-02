# morie.fn -- function file (rootcoder007/morie)
"""Gini's gamma from a copula."""

from . import _array_core as np

from ._copula import FAMILIES, copula_cdf
from ._richresult import RichResult

__all__ = ["ginis_gamma_copula"]


def ginis_gamma_copula(family, theta=None, nu=None, n=400):
    r"""Gini's gamma, the diagonal-section dependence measure.

    .. math:: \gamma = 4\left[\int_0^1 C(u, 1-u)\,du
              - \int_0^1 \big(u - C(u, u)\big)du\right],

    contrasting the copula's two diagonals: the first integral is
    large under negative dependence (mass on the anti-diagonal), the
    second under positive dependence. gamma is 0 for independence,
    +1 for comonotonicity and -1 for countermonotonicity.

    Parameters
    ----------
    family : str
        Copula family.
    theta : float
        Parameter.
    nu : float, optional
        t degrees of freedom.
    n : int, default 400
        Quadrature points per diagonal.

    Returns
    -------
    RichResult
        keys: ``gamma``, ``anti_diagonal``, ``diagonal_gap``,
        ``family``, ``theta``, ``method``.

    References
    ----------
    Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.).
    Springer. Sec. 5.1.4 (Gini's gamma).

    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Ch. 2.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    n = int(n)
    if n < 20:
        raise ValueError(f"n must be at least 20, got {n}.")
    u = (np.arange(n) + 0.5) / n
    anti = float(np.mean(copula_cdf(family, u, 1.0 - u, theta, nu)))
    diag = float(np.mean(u - copula_cdf(family, u, u, theta, nu)))
    return RichResult(
        payload={
            "gamma": 4.0 * (anti - diag),
            "anti_diagonal": anti,
            "diagonal_gap": diag,
            "family": family,
            "theta": None if theta is None else float(theta),
            "method": "Gini's gamma = 4[int C(u, 1-u) du - int (u - C(u, u)) du]",
        }
    )


def cheatsheet():
    return "ginicop: gamma contrasts the two diagonals; 0 independence, +1 comonotone"
