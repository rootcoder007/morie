# morie.fn -- function file (rootcoder007/morie)
"""Moments of the r-th order statistic from a Uniform(0,1) parent."""

import math

from ._richresult import RichResult

__all__ = ['ostatmom', 'gibbons_order_moments']


def ostatmom(r, n, k=1):
    """k-th raw moment of U_(r), plus its mean and variance.

    Section 2.4 (book p. 38).  Because U_(r) ~ Beta(r, n-r+1),

    .. math::
        E[U_{(r)}^k] = \\prod_{j=0}^{k-1} \\frac{r+j}{n+1+j},
        \\qquad E[U_{(r)}] = \\frac{r}{n+1},
        \\qquad Var[U_{(r)}] = \\frac{r(n-r+1)}{(n+1)^2(n+2)}.

    Parameters
    ----------
    r, n : int
        Order-statistic index and sample size, 1 <= r <= n.
    k : int, optional
        Moment order, k >= 1 (default 1).

    Returns
    -------
    RichResult
        keys ``moment``, ``mean``, ``var``, ``r``, ``n``, ``k``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 2.4, p. 38 (Beta moments of
    the uniform order statistics).
    """
    r = int(r)
    n = int(n)
    k = int(k)
    if not 1 <= r <= n:
        raise ValueError("need 1 <= r <= n.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    mom = 1.0
    for j in range(k):
        mom *= (r + j) / (n + 1 + j)
    return RichResult(
        payload={
            "moment": float(mom),
            "mean": r / (n + 1.0),
            "var": r * (n - r + 1.0) / ((n + 1.0) ** 2 * (n + 2.0)),
            "r": r,
            "n": n,
            "k": k,
            "method": "E[U_(r)^k] = prod_{j<k} (r+j)/(n+1+j)",
        }
    )


gibbons_order_moments = ostatmom
