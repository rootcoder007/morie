# morie.fn -- function file (rootcoder007/morie)
"""Bivariate logistic max-stable distribution."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_max_stable_logistic"]


def evt_max_stable_logistic(x, y, alpha):
    """
    Bivariate logistic max-stable distribution

    Formula: F(x,y) = exp(-((x^(-1/alpha) + y^(-1/alpha))^alpha))

    Coles (2001) eq. (8.10), p. 146, on unit Frechet margins.  As
    alpha -> 1 the exponent becomes 1/x + 1/y and the margins are
    independent; as alpha -> 0 it becomes max(1/x, 1/y) and they are
    perfectly dependent.  The Pickands function of this family is
    A(t) = (t^(1/alpha) + (1-t)^(1/alpha))^alpha, and the coefficient of
    upper tail dependence is 2 - 2^alpha.

    Parameters
    ----------
    x : array-like
        First coordinate, strictly positive (unit Frechet scale).
    y : array-like
        Second coordinate, strictly positive.
    alpha : float
        Dependence parameter in (0, 1].

    Returns
    -------
    result : dict
        Keys: F, estimate, V, A_half, chi, n.

    References
    ----------
    Tawn (1988), Biometrika 75(3):397-415.
    Coles (2001), An Introduction to Statistical Modeling of Extreme
    Values, Springer, eq. (8.10) p. 146.
    """
    xs = core.vec(x)
    ys = core.vec(y)
    alpha = float(alpha)
    if not xs or not ys:
        raise ValueError("empty input: x and y are required")
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if any(v <= 0.0 for v in xs) or any(v <= 0.0 for v in ys):
        raise ValueError("x and y must be strictly positive on the Frechet scale")
    F, V = [], []
    for i in range(len(xs)):
        v = (xs[i] ** (-1.0 / alpha) + ys[i] ** (-1.0 / alpha)) ** alpha
        V.append(v)
        F.append(math.exp(-v))
    a_half = (0.5 ** (1.0 / alpha) + 0.5 ** (1.0 / alpha)) ** alpha
    return RichResult(payload={
        "F": F,
        "estimate": F[0],
        "V": V,
        "A_half": a_half,
        "chi": 2.0 - 2.0 ** alpha,
        "n": len(xs),
        "method": "bivariate logistic max-stable distribution",
    })


def cheatsheet():
    return "evmsexp: bivariate logistic max-stable distribution"


# compact alias per ledger/NAMING.md
evtmaxstablelogistic = evt_max_stable_logistic
