# morie.fn -- function file (rootcoder007/morie)
"""Azuma-Hoeffding concentration bound for martingales."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['azuma', 'martingale_concentration']


def azuma(c, t):
    """Azuma-Hoeffding concentration bound for martingales.

    The bound needs only bounded differences, not independence, which is why it survives where Hoeffding's own inequality does not. Both the one-sided and the two-sided (doubled) bound are returned; the two-sided one is the one people usually want and the one they usually forget to double.


    Formula: P(M_n - M_0 >= t) <= exp(-t^2 / (2 sum_i c_i^2))

    Parameters
    ----------
    c : array-like
        Bounded-difference constants c_i, one per step.
    t : float
        Deviation whose probability is bounded.

    Returns
    -------
    RichResult
        ``bound`` (one-sided), ``bound_two_sided``, ``sum_c2``, ``t``, ``n``.

    References
    ----------
    Azuma (1967), Weighted sums of certain dependent random variables,
    Tohoku Mathematical Journal 19:357-367; Hoeffding (1963), JASA
    58:13-30.  Neither is held locally; the inequality is stated in this
    exact form in every standard reference consulted.
    """
    c = C.vec(c)
    t = float(t)
    if any(v < 0 for v in c):
        raise ValueError("bounded-difference constants must be non-negative")
    s = sum(v * v for v in c)
    if s <= 0:
        raise ValueError("sum of squared differences must be positive")
    b = math.exp(-t * t / (2.0 * s))
    return RichResult(payload={
        "bound": b, "bound_two_sided": min(1.0, 2.0 * b), "sum_c2": s,
        "t": t, "n": len(c), "method": "Azuma-Hoeffding bound"})


martingale_concentration = azuma


def cheatsheet():
    return "mrgdrv: Azuma-Hoeffding concentration bound for martingales."
