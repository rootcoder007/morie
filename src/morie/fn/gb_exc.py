# morie.fn -- function file (rootcoder007/morie)
"""Null distribution of the placement/exceedance statistic P_(i)."""

import math

from ._richresult import RichResult

__all__ = ['exceed', 'gibbons_exceedance_stat']


def exceed(i, m, n, j=None):
    """Exact null law of the placement P_(i) = m S_m(Y_(i)).

    Problem 2.28(c) (book p. 70): with two independent samples of
    sizes m (the X's) and n (the Y's) from the same continuous cdf,

    .. math::
        P[P_{(i)} = j] = \\frac{\\binom{m+n-i-j}{m-j}
            \\binom{i+j-1}{j}}{\\binom{m+n}{n}},
        \\qquad j = 0, 1, \\dots, m.

    The exceedance statistic m - P_(i) counts the X's that exceed
    Y_(i); its pmf is the same vector read backwards.

    Parameters
    ----------
    i : int
        Index of the Y order statistic, 1 <= i <= n.
    m, n : int
        Sizes of the X and Y samples.
    j : int, optional
        Placement value at which to report the pmf and cdf.

    Returns
    -------
    RichResult
        keys ``pmf`` (list over j = 0..m), ``pmf_j``, ``cdf_j``,
        ``mean``, ``var``, ``i``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Problem 2.28(c), p. 70.
    """
    i = int(i)
    m = int(m)
    n = int(n)
    if not 1 <= i <= n:
        raise ValueError("need 1 <= i <= n.")
    if m < 1:
        raise ValueError("m must be at least 1.")
    den = math.comb(m + n, n)
    pmf = [
        math.comb(m + n - i - k, m - k) * math.comb(i + k - 1, k) / den
        for k in range(m + 1)
    ]
    mean = sum(k * pk for k, pk in enumerate(pmf))
    ex2 = sum(k * k * pk for k, pk in enumerate(pmf))
    out = {
        "pmf": pmf,
        "pmf_j": float("nan"),
        "cdf_j": float("nan"),
        "mean": float(mean),
        "var": float(ex2 - mean * mean),
        "i": i,
        "m": m,
        "n": n,
        "method": "P[P_(i)=j] = C(m+n-i-j, m-j) C(i+j-1, j) / C(m+n, n)",
    }
    if j is not None:
        j = int(j)
        if not 0 <= j <= m:
            raise ValueError("j must lie in 0..m.")
        out["pmf_j"] = pmf[j]
        out["cdf_j"] = float(sum(pmf[: j + 1]))
    return RichResult(payload=out)


gibbons_exceedance_stat = exceed
