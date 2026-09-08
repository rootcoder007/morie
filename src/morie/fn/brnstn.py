# morie.fn -- function file (rootcoder007/morie)
"""Bernstein's concentration inequality."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['bernstein', 'bernstein_inequality']


def bernstein(sigma2, M, n, t):
    """Bernstein's concentration inequality.

    Bernstein beats Hoeffding whenever the variance is small relative to the range: the denominator is variance-driven for small t and range-driven only in the large-deviation tail. The ratio to the Hoeffding bound is returned so that the crossover is visible rather than asserted.


    Formula: P(S_n >= n t) <= exp(-n t^2 / (2 sigma^2 + 2 M t / 3))

    Parameters
    ----------
    sigma2 : float
        Bound on the per-summand variance.
    M : float
        Almost-sure bound on |X_i - E X_i|.
    n : int
        Number of summands.
    t : float
        Deviation per summand.

    Returns
    -------
    RichResult
        ``bound``, ``bound_two_sided``, ``hoeffding``, ``ratio``, ``exponent``.

    References
    ----------
    Bernstein (1924).  The original is not held locally and is in
    Russian; the inequality is stated in this exact form in every standard
    concentration-inequality reference consulted.
    """
    s2 = float(sigma2); M = float(M); n = int(n); t = float(t)
    if s2 < 0 or M < 0 or n < 1 or t < 0:
        raise ValueError("need sigma2 >= 0, M >= 0, n >= 1, t >= 0")
    den = 2.0 * s2 + 2.0 * M * t / 3.0
    if den <= 0:
        raise ValueError("degenerate bound: sigma2 and M are both zero")
    ex = -n * t * t / den
    b = math.exp(ex)
    hoef = math.exp(-n * t * t / (2.0 * M * M)) if M > 0 else 0.0
    return RichResult(payload={
        "bound": b, "bound_two_sided": min(1.0, 2.0 * b),
        "hoeffding": hoef, "ratio": b / hoef if hoef > 0 else float("inf"),
        "exponent": ex, "method": "Bernstein inequality"})


bernstein_inequality = bernstein


def cheatsheet():
    return "brnstn: Bernstein's concentration inequality."
