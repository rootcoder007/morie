# morie.fn -- function file (rootcoder007/morie)
"""Sample p-th quantile as an order statistic point estimator."""

import math

from ._richresult import RichResult

__all__ = ['sampquant', 'gibbons_marginal_quant']


def sampquant(x, p):
    """Point estimate of the population p-th quantile.

    Section 2.6 (book p. 42): the sample quantile of order p is the
    order statistic X_(r) with

    .. math:: r = \\lfloor np \\rfloor + 1,

    i.e. the smallest order statistic whose empirical cdf value
    reaches p.  Its exact mean and variance are the Beta moments of
    Theorem 2.4.3 pushed through F^{-1}; the uniform-scale moments are
    returned here as ``u_mean`` and ``u_var``.

    Parameters
    ----------
    x : sequence of float
        The sample, n >= 1.
    p : float
        Quantile level, 0 < p < 1.

    Returns
    -------
    RichResult
        keys ``estimate``, ``r``, ``n``, ``p``, ``u_mean``, ``u_var``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 2.6, p. 42.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    p = float(p)
    if n < 1:
        raise ValueError("x must be non-empty.")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1).")
    r = int(math.floor(n * p)) + 1
    if r > n:
        r = n
    return RichResult(
        payload={
            "estimate": xs[r - 1],
            "r": r,
            "n": n,
            "p": p,
            "u_mean": r / (n + 1.0),
            "u_var": r * (n - r + 1.0) / ((n + 1.0) ** 2 * (n + 2.0)),
            "method": "sample quantile X_(floor(np)+1) (Gibbons Sec. 2.6)",
        }
    )


gibbons_marginal_quant = sampquant
