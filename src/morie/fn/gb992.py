# morie.fn -- function file (rootcoder007/morie)
"""Rosenbaum outside-the-extremes scale test -- eq. (9.9.2)."""

import math

from ._richresult import RichResult

__all__ = ['rosenbm', 'gibbons_fligner_killeen']


def rosenbm(x, y):
    """R = number of X's outside the range of the Y sample.

    Section 9.9 (book p. 329), eq. (9.9.2).  Assuming the two
    populations share a location, Rosenbaum (1953) counts the X
    observations that are either smaller than the smallest Y or larger
    than the largest Y.  Under H0

    .. math:: f_R(r) = n(n-1)\\binom{m}{r} B(m+n-1-r,\\; r+2),

    with B the beta function; large R is evidence that the X sample is
    the more widely dispersed.

    NOTE ON THE MODULE LABEL: the generated stub called this a
    "Fligner-Killeen test".  Fligner & Killeen (1976) is not cited
    anywhere in Gibbons & Chakraborti (2011) -- the only Fligner
    reference in the book is Fligner and Wolfe (1976), on placements --
    so this module implements what the cited source actually gives at
    that point, Rosenbaum's test.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``statistic`` (R), ``p_value`` (upper tail), ``pmf``,
        ``ymin``, ``ymax``, ``mean``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.9, eq. (9.9.2), p. 329
    (Rosenbaum, 1953); verification is Problem 9.9, p. 341.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 2:
        raise ValueError("need m >= 1 and n >= 2.")
    ymin = min(ys)
    ymax = max(ys)
    r = sum(1 for v in xs if v < ymin or v > ymax)

    def _beta(a, b):
        return math.exp(
            math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        )

    pmf = [
        n * (n - 1.0) * math.comb(m, k) * _beta(m + n - 1.0 - k, k + 2.0)
        for k in range(m + 1)
    ]
    mean = sum(k * p for k, p in enumerate(pmf))
    return RichResult(
        payload={
            "statistic": int(r),
            "p_value": float(min(1.0, sum(pmf[r:]))),
            "pmf": pmf,
            "ymin": float(ymin),
            "ymax": float(ymax),
            "mean": float(mean),
            "m": m,
            "n": n,
            "method": "Rosenbaum outside-extremes scale test, eq. (9.9.2)",
        }
    )


gibbons_fligner_killeen = rosenbm
