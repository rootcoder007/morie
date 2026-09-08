# morie.fn -- function file (rootcoder007/morie)
"""Rank-order statistics: midranks of the absolute differences."""

import math

from ._richresult import RichResult

__all__ = ['absrank', 'gibbons_rank_order_stat']


def absrank(d):
    """Ranks of |d_i| among the absolute values, with their signs.

    Section 5.5 (book p. 189).  Ties get midranks (Sec. 5.6.2), which
    is the convention the signed-rank test uses.  The sign attached to
    each rank is the sign of the original difference.

    Parameters
    ----------
    d : sequence of float
        Differences, n >= 1.

    Returns
    -------
    RichResult
        keys ``ranks`` (midranks of |d|), ``signs``, ``signed``
        (sign * rank), ``ties`` (multiplicities of tied |d| groups),
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.5, p. 189; midranks Sec. 5.6.2.
    """
    ds = [float(v) for v in d]
    n = len(ds)
    if n < 1:
        raise ValueError("d must be non-empty.")
    a = [abs(v) for v in ds]
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    ties = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    signs = [(1 if v > 0 else (-1 if v < 0 else 0)) for v in ds]
    return RichResult(
        payload={
            "ranks": ranks,
            "signs": signs,
            "signed": [signs[i] * ranks[i] for i in range(n)],
            "ties": ties,
            "n": n,
            "method": "midranks of |d_i| with original signs (Sec. 5.5)",
        }
    )


gibbons_rank_order_stat = absrank
