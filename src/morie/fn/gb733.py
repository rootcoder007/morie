# morie.fn -- function file (rootcoder007/morie)
"""Covariance of two linear rank statistics -- Theorem 7.3.3."""

import math

from ._richresult import RichResult

__all__ = ['lrankcov', 'gibbons_linrank_covariance']


def lrankcov(a, b, m, n):
    """Cov(B_N, T_N) for two linear rank statistics on the same ranking.

    Theorem 7.3.3 (book p. 279):

    .. math:: Cov(B_N, T_N) = \\frac{mn}{N^2(N-1)}
        \\left[N\\sum_i a_i b_i
              - \\sum_i a_i \\sum_i b_i\\right],

    under H0: F_X = F_Y.  Taking b = a recovers the variance of
    Theorem 7.3.2, which is returned as ``var_a`` and ``var_b`` so the
    correlation can be read off directly.

    Parameters
    ----------
    a, b : sequence of float
        The two score vectors, both of length N = m + n.
    m, n : int
        The two sample sizes.

    Returns
    -------
    RichResult
        keys ``cov``, ``corr``, ``var_a``, ``var_b``, ``N``, ``m``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 7.3.3, p. 279.
    """
    av = [float(v) for v in a]
    bv = [float(v) for v in b]
    m = int(m)
    n = int(n)
    nn = m + n
    if len(av) != nn or len(bv) != nn:
        raise ValueError("score vectors must have length m + n.")
    if nn < 2:
        raise ValueError("N must be at least 2.")
    k = m * n / (float(nn) ** 2 * (nn - 1.0))
    cov = k * (nn * sum(av[i] * bv[i] for i in range(nn)) - sum(av) * sum(bv))
    va = k * (nn * sum(v * v for v in av) - sum(av) ** 2)
    vb = k * (nn * sum(v * v for v in bv) - sum(bv) ** 2)
    corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
    return RichResult(
        payload={
            "cov": float(cov),
            "corr": float(corr),
            "var_a": float(va),
            "var_b": float(vb),
            "N": int(nn),
            "m": m,
            "n": n,
            "method": "Cov(B_N, T_N), Gibbons Theorem 7.3.3",
        }
    )


gibbons_linrank_covariance = lrankcov
