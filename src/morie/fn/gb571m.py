# morie.fn -- function file (rootcoder007/morie)
"""Null mean and variance of the signed-rank statistic T+."""

import math

from ._richresult import RichResult

__all__ = ['wsrmom', 'gibbons_wsrt_mean']


def wsrmom(n):
    """Moments of T+ under H0 -- Gibbons eq. (5.7.2).

    Book p. 197:

    .. math:: E[T^+] = \\frac{N(N+1)}{4}, \\qquad
        Var[T^+] = \\frac{N(N+1)(2N+1)}{24}.

    The third central moment is 0: the null distribution of T+ is
    symmetric about N(N+1)/4 (book p. 200), which is why one tail of
    Table H suffices.

    Parameters
    ----------
    n : int
        Number of non-zero differences, n >= 1.

    Returns
    -------
    RichResult
        keys ``mean``, ``var``, ``sd``, ``total`` (N(N+1)/2),
        ``skew``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (5.7.2), p. 197.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1.")
    mean = n * (n + 1.0) / 4.0
    var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    return RichResult(
        payload={
            "mean": float(mean),
            "var": float(var),
            "sd": float(math.sqrt(var)),
            "total": float(n * (n + 1.0) / 2.0),
            "skew": 0.0,
            "n": n,
            "method": "E[T+] = N(N+1)/4, Var[T+] = N(N+1)(2N+1)/24",
        }
    )


gibbons_wsrt_mean = wsrmom
