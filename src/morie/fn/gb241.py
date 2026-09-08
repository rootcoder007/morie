# morie.fn -- function file (rootcoder007/morie)
"""CDF of the r-th order statistic as a binomial tail -- Theorem 2.4.1."""

import math

from ._richresult import RichResult

__all__ = ['ostatcdf', 'gibbons_order_cdf']


def ostatcdf(t, r, n, cdf):
    """P[X_(r) <= t] as an upper binomial tail in F_X(t).

    Theorem 2.4.1 (book p. 37), eq. (2.4.1):

    .. math::
        P[X_{(r)} \\le t] = \\sum_{i=r}^{n} \\binom{n}{i}
            [F_X(t)]^i [1 - F_X(t)]^{n-i}.

    Parameters
    ----------
    t : float
        Argument of the cdf.
    r : int
        Order-statistic index, 1 <= r <= n.
    n : int
        Sample size.
    cdf : callable or float
        Either F_X (a callable) or the number F_X(t) itself.

    Returns
    -------
    RichResult
        keys ``cdf`` (the probability), ``fx`` (F_X(t)), ``r``, ``n``,
        ``sf``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 2.4.1, eq. (2.4.1), p. 37.
    """
    r = int(r)
    n = int(n)
    if not 1 <= r <= n:
        raise ValueError("need 1 <= r <= n.")
    p = float(cdf(t)) if callable(cdf) else float(cdf)
    if not 0.0 <= p <= 1.0:
        raise ValueError("F_X(t) must lie in [0, 1].")
    val = sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(r, n + 1))
    return RichResult(
        payload={
            "cdf": float(val),
            "sf": float(1.0 - val),
            "fx": p,
            "r": r,
            "n": n,
            "t": float(t),
            "method": "P[X_(r) <= t] = sum_{i>=r} C(n,i) F^i (1-F)^(n-i)",
        }
    )


gibbons_order_cdf = ostatcdf
