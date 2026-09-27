"""Generalized lambda distribution (Ramberg-Schmeiser parameterization), defined by its quantile function.

Ramberg, J. S. & Schmeiser, B. W. (1974). An approximate method for generating asymmetric random variables.
Communications of the ACM 17, 78-82.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult

__all__ = ["genlambda"]


def _u_of_x(Q, x, lo, hi):
    """Solve Q(u) = x for u in (0, 1) by bisection (Q increasing)."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    a, b = 0.0, 1.0
    for _ in range(200):
        m = (a + b) / 2
        if Q(m) < x:
            a = m
        else:
            b = m
        if b - a < 1e-16:
            break
    return (a + b) / 2


def genlambda(x=None, l1=0.0, l2=1.0, l3=0.1, l4=0.1, p=None, n=0, seed=0):
    r"""Q(u) = l1 + (u^l3 - (1 - u)^l4)/l2, with density 1/Q'(u) at x = Q(u).

    Q'(u) = (l3 u^(l3-1) + l4 (1-u)^(l4-1))/l2 must be positive on (0, 1). The
    distribution function at x is the u solving Q(u) = x (bisection); draws are
    Q of morie's Philox uniforms (identical in the R arm). l1 = 0 and
    l2 = l3 = l4 = lambda gives Tukey's lambda distribution.

    Parameters
    ----------
    x : float or sequence, optional
    l1, l2, l3, l4 : float
        Location, scale and the two shape parameters.
    p : float or sequence, optional
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, cdf, quantile, random, support.

    References
    ----------
    Ramberg, J. S. & Schmeiser, B. W. (1974). Communications of the ACM 17, 78-82.
    Matches ``gld::dgl(x, l1, l2, l3, l4, param = "rs")``.

    Examples
    --------
    >>> genlambda(p=0.5, l1=1.0, l2=0.2, l3=0.13, l4=0.13)["quantile"]
    1.0
    """
    if l2 == 0:
        raise ValueError("l2 must be non-zero")

    def Q(u):
        return l1 + (u**l3 - (1 - u) ** l4) / l2

    def dQ(u):
        return (l3 * u ** (l3 - 1) + l4 * (1 - u) ** (l4 - 1)) / l2

    for u in (0.001, 0.25, 0.5, 0.75, 0.999):
        if not dQ(u) > 0:
            raise ValueError("invalid parameters: Q'(u) must be positive on (0, 1)")
    lo = Q(0.0) if l3 > 0 else -math.inf
    hi = Q(1.0) if l4 > 0 else math.inf

    def cdf(v):
        return _u_of_x(Q, v, lo, hi)

    def pdf(v):
        if v <= lo or v >= hi:
            return 0.0
        return 1 / dQ(cdf(v))

    payload = {"support": (lo, hi)}
    if x is not None:
        xs, sc = vec(x)
        dens = [pdf(v) for v in xs]
        payload["pdf"] = out(dens, sc)
        payload["logpdf"] = out([math.log(v) if v > 0 else -math.inf for v in dens], sc)
        payload["cdf"] = out([cdf(v) for v in xs], sc)
    if p is not None:
        ps, sc = vec(p)
        payload["quantile"] = out([Q(u) for u in ps], sc)
    if n:
        payload["random"] = draws(Q, n, seed)
    return RichResult(title="Generalized lambda distribution", summary_lines=[("support", (lo, hi))], payload=payload)


def cheatsheet():
    return "genlambda: generalized lambda (Ramberg-Schmeiser) by its quantile function; pdf 1/Q'(u)."
