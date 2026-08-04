# morie.fn -- function file (rootcoder007/morie)
"""Distribution-free confidence interval for a population quantile."""

import math

from ._richresult import RichResult

__all__ = ['quantci', 'gibbons_quantile_ci']


def quantci(x, p, r, s):
    """Order-statistic confidence interval (X_(r), X_(s)) for x_p.

    Section 5.2 (book p. 158): for a continuous parent the interval
    (X_(r), X_(s)) covers the population p-th quantile with the
    distribution-free probability

    .. math::
        P[X_{(r)} < x_p < X_{(s)}] = \\sum_{i=r}^{s-1}
            \\binom{n}{i} p^i (1-p)^{n-i},

    which depends on n, r, s and p only -- never on F.

    Parameters
    ----------
    x : sequence of float
        The sample, n >= 2.
    p : float
        Quantile level, 0 < p < 1.
    r, s : int
        Order-statistic indices, 1 <= r < s <= n.

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``coverage``, ``alpha``, ``r``,
        ``s``, ``n``, ``p``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.2, p. 158.
    """
    xs = sorted(float(v) for v in x)
    n = len(xs)
    r = int(r)
    s = int(s)
    p = float(p)
    if n < 2:
        raise ValueError("need at least 2 observations.")
    if not 1 <= r < s <= n:
        raise ValueError("need 1 <= r < s <= n.")
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1).")
    cov = sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(r, s))
    return RichResult(
        payload={
            "lower": xs[r - 1],
            "upper": xs[s - 1],
            "coverage": float(cov),
            "alpha": float(1.0 - cov),
            "r": r,
            "s": s,
            "n": n,
            "p": p,
            "method": "P[X_(r) < x_p < X_(s)] = sum_{i=r}^{s-1} C(n,i) p^i q^(n-i)",
        }
    )


gibbons_quantile_ci = quantci
