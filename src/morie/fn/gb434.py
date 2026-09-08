# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of the one-sided KS statistic D+."""

import math

from ._richresult import RichResult

__all__ = ['ksplusdist', 'gibbons_ks_one_sided_dist']


def ksplusdist(c, n):
    """P(D+_n >= c) by the Birnbaum-Tingey closed form.

    Theorem 4.3.4 (book p. 115) expresses P(D+_n < c) as a nested
    integral over the uniform order statistics; carrying the
    integration out gives the finite sum

    .. math:: P(D_n^+ \\ge c) = c \\sum_{j=0}^{\\lfloor n(1-c)\\rfloor}
        \\binom{n}{j}\\left(c + \\frac{j}{n}\\right)^{j-1}
        \\left(1 - c - \\frac{j}{n}\\right)^{n-j},

    which is exact for every n and needs no quadrature.

    Parameters
    ----------
    c : float
        Value of the one-sided statistic, 0 < c < 1.
    n : int
        Sample size.

    Returns
    -------
    RichResult
        keys ``sf`` (P(D+ >= c)), ``cdf``, ``terms``, ``n``, ``c``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 4.3.4, p. 115
    (Birnbaum and Tingey, 1951).
    """
    c = float(c)
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1.")
    if c <= 0.0:
        return RichResult(
            payload={"sf": 1.0, "cdf": 0.0, "terms": 0, "n": n, "c": c,
                     "method": "P(D+ >= c), Birnbaum-Tingey (Thm 4.3.4)"}
        )
    if c >= 1.0:
        return RichResult(
            payload={"sf": 0.0, "cdf": 1.0, "terms": 0, "n": n, "c": c,
                     "method": "P(D+ >= c), Birnbaum-Tingey (Thm 4.3.4)"}
        )
    jmax = int(math.floor(n * (1.0 - c)))
    total = 0.0
    for j in range(jmax + 1):
        total += (
            math.comb(n, j)
            * (c + j / n) ** (j - 1)
            * (1.0 - c - j / n) ** (n - j)
        )
    sf = min(1.0, max(0.0, c * total))
    return RichResult(
        payload={
            "sf": float(sf),
            "cdf": float(1.0 - sf),
            "terms": int(jmax + 1),
            "n": n,
            "c": c,
            "method": "P(D+ >= c), Birnbaum-Tingey (Thm 4.3.4)",
        }
    )


gibbons_ks_one_sided_dist = ksplusdist
