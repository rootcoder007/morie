"""Tukey's lambda distribution, defined by its quantile function.

Tukey, J. W. (1960). The practical relationship between the common transformations of percentages or fractions
and of amounts. Technical Report 36, Statistical Research Group, Princeton University.
"""

import math

from ._distcore import draws, out, vec
from ._richresult import RichResult
from .genlambda import _u_of_x

__all__ = ["tukeylambda"]


def tukeylambda(x=None, lam=0.14, p=None, n=0, seed=0):
    r"""Q(u) = (u^lam - (1 - u)^lam)/lam (lam != 0), the logistic log(u/(1-u)) at lam = 0.

    The density at x = Q(u) is 1/Q'(u) with Q'(u) = u^(lam-1) + (1-u)^(lam-1)
    (1/(u(1-u)) at lam = 0); the support is [-1/lam, 1/lam] for lam > 0. lam = 0.14
    approximates the normal, lam = 1 and 2 give uniform distributions.

    Parameters
    ----------
    x : float or sequence, optional
    lam : float
    p : float or sequence, optional
    n : int
    seed : int

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, cdf, quantile, random, support.

    References
    ----------
    Tukey, J. W. (1960). Princeton Statistical Research Group Technical Report 36.
    Joiner, B. L. & Rosenblatt, J. R. (1971). JASA 66, 394-399.

    Examples
    --------
    >>> round(tukeylambda(0.25, lam=1.0)["cdf"], 12)
    0.625
    """
    if lam == 0:

        def Q(u):
            if u <= 0:
                return -math.inf
            if u >= 1:
                return math.inf
            return math.log(u / (1 - u))

        def dQ(u):
            return 1 / (u * (1 - u))
    else:

        def Q(u):
            return (u**lam - (1 - u) ** lam) / lam

        def dQ(u):
            return u ** (lam - 1) + (1 - u) ** (lam - 1)

    lo = -1 / lam if lam > 0 else -math.inf
    hi = 1 / lam if lam > 0 else math.inf

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
    return RichResult(title="Tukey lambda distribution", summary_lines=[("lambda", lam)], payload=payload)


def cheatsheet():
    return "tukeylambda: Tukey lambda distribution by its quantile function (u^l - (1-u)^l)/l."
