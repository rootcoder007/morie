# morie.fn -- function file (rootcoder007/morie)
"""Westenberg interquartile-range scale test -- eq. (9.9.1)."""

import math

from ._richresult import RichResult

__all__ = ['wstnbrg', 'gibbons_conover_scale']


def wstnbrg(x, y):
    """U = number of X observations inside the Y interquartile range.

    Section 9.9 (book p. 329), eq. (9.9.1).  Two populations differing
    only in scale put unequal proportions of their samples between two
    symmetric quantile points of the combined sample.  Taking those
    quantiles to be the quartiles, U is the count of X's inside the
    Y-sample interquartile range and, when N = m + n is divisible by 4
    so no observation coincides with a quartile, U is hypergeometric:

    .. math:: f_U(u) = \\frac{\\binom{m}{u}\\binom{n}{N/2 - u}}
        {\\binom{N}{N/2}}.

    Small U says the X's are the more widely dispersed sample, so the
    rejection region for that alternative is the lower tail.

    NOTE ON THE MODULE LABEL: the generated stub called this a
    "Conover squared ranks test".  The phrase "squared ranks" does not
    occur anywhere in Gibbons & Chakraborti (2011), and Conover's
    squared-ranks test is not in the book; Sec. 9.9 -- the only place
    the cited source treats further scale tests -- gives Westenberg's
    test and Rosenbaum's.  This module implements the cited source.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``statistic`` (U), ``p_value`` (lower tail), ``pmf``,
        ``q1``, ``q3`` (the Y quartiles used), ``mean``, ``var``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.9, eq. (9.9.1), p. 329
    (Westenberg, 1948).
    """
    xs = [float(v) for v in x]
    ys = sorted(float(v) for v in y)
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 2:
        raise ValueError("need m >= 1 and n >= 2.")
    nn = m + n

    def _q(p):
        h = (n - 1) * p
        lo = int(math.floor(h))
        hi = min(lo + 1, n - 1)
        return ys[lo] + (h - lo) * (ys[hi] - ys[lo])

    q1 = _q(0.25)
    q3 = _q(0.75)
    u = sum(1 for v in xs if q1 <= v <= q3)
    half = nn // 2
    den = math.comb(nn, half)
    pmf = [
        (
            math.comb(m, k) * math.comb(n, half - k) / den
            if 0 <= half - k <= n
            else 0.0
        )
        for k in range(m + 1)
    ]
    mean = m * half / float(nn)
    var = m * n * half * (nn - half) / (float(nn) ** 2 * (nn - 1.0))
    return RichResult(
        payload={
            "statistic": int(u),
            "p_value": float(min(1.0, sum(pmf[: u + 1]))),
            "pmf": pmf,
            "q1": float(q1),
            "q3": float(q3),
            "mean": float(mean),
            "var": float(var),
            "m": m,
            "n": n,
            "method": "Westenberg interquartile scale test, eq. (9.9.1)",
        }
    )


gibbons_conover_scale = wstnbrg
