# morie.fn -- function file (rootcoder007/morie)
"""Husler-Reiss bivariate extreme-value dependence."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_husler_reiss_dep"]


def evt_husler_reiss_dep(x, y, lam):
    """
    Husler-Reiss bivariate dependence

    Formula: F(x,y) = exp(-x Phi(lam + log(y/x)/(2 lam))
                          - y Phi(lam + log(x/y)/(2 lam)))

    Written on the standard exponential scale, where each margin is
    exp(-x).  As lam -> 0 the exponent tends to min(x, y), perfect
    dependence; as lam -> infinity it tends to x + y, independence.  The
    Pickands function is
    A(t) = t Phi(lam + log((1-t)/t)/(2 lam)) + (1-t) Phi(lam + log(t/(1-t))/(2 lam)).

    Parameters
    ----------
    x : array-like
        First coordinate, strictly positive.
    y : array-like
        Second coordinate, strictly positive.
    lam : float
        Dependence parameter, strictly positive.

    Returns
    -------
    result : dict
        Keys: F, estimate, V, A_half, chi, n.

    References
    ----------
    Husler & Reiss (1989), Statist. Probab. Letters 7(4):283-286.
    """
    xs = core.vec(x)
    ys = core.vec(y)
    lam = float(lam)
    if not xs or not ys:
        raise ValueError("empty input: x and y are required")
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if not (lam > 0.0):
        raise ValueError("lam must be strictly positive")
    if any(v <= 0.0 for v in xs) or any(v <= 0.0 for v in ys):
        raise ValueError("x and y must be strictly positive")
    F, V = [], []
    for i in range(len(xs)):
        a = xs[i]
        b = ys[i]
        v = a * core.pnorm(lam + math.log(b / a) / (2.0 * lam)) \
            + b * core.pnorm(lam + math.log(a / b) / (2.0 * lam))
        V.append(v)
        F.append(math.exp(-v))
    a_half = core.pnorm(lam)
    return RichResult(payload={
        "F": F,
        "estimate": F[0],
        "V": V,
        "A_half": a_half,
        "chi": 2.0 - 2.0 * core.pnorm(lam),
        "n": len(xs),
        "method": "Husler-Reiss bivariate extreme-value dependence",
    })


def cheatsheet():
    return "evhrid: Husler-Reiss bivariate dependence"


# compact alias per ledger/NAMING.md
evthuslerreissdep = evt_husler_reiss_dep
