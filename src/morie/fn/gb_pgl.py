# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of Page's L by enumeration."""

import math

from ._richresult import RichResult

__all__ = ['pageexact', 'gibbons_page_exact']


def pageexact(k, n, ell=None):
    """Exact null law of L, built block by block.

    Section 12.3 (book p. 448); Table Q, p. 591, prints selected
    critical values.  Under H0 the n! rankings within a block are
    equally likely and blocks are independent, so the distribution of
    L = sum_j j R_j is the k-fold convolution of the within-block
    distribution of sum_j j r_j over the n! permutations.  Enumerating
    one block and convolving k times is exact and far cheaper than
    enumerating (n!)^k tables.

    Parameters
    ----------
    k : int
        Number of blocks, k >= 1.
    n : int
        Number of treatments, 2 <= n <= 8 (n! enumeration).
    ell : float, optional
        Value of L at which to report the pmf and the upper tail.

    Returns
    -------
    RichResult
        keys ``support``, ``pmf``, ``pmf_l``, ``sf_l``, ``mean``,
        ``var``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 12.3, p. 448; Table Q, p. 591.
    """
    k = int(k)
    n = int(n)
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not 2 <= n <= 8:
        raise ValueError("n must lie in 2..8 for exact enumeration.")
    lo = sum((j + 1) * (n - j) for j in range(n))
    hi = sum((j + 1) * (j + 1) for j in range(n))
    span = hi - lo
    block = [0.0] * (span + 1)
    perm = list(range(1, n + 1))

    def _rec(rem, used, acc, pos):
        if pos == n:
            block[acc - lo] += 1.0
            return
        for v in rem:
            _rec([t for t in rem if t != v], used, acc + (pos + 1) * v, pos + 1)

    _rec(perm, None, 0, 0)
    tot = sum(block)
    block = [c / tot for c in block]
    cur = [1.0]
    for _ in range(k):
        new = [0.0] * (len(cur) + span)
        for i, a in enumerate(cur):
            if a == 0.0:
                continue
            for j, b in enumerate(block):
                new[i + j] += a * b
        cur = new
    support = [lo * k + i for i in range(len(cur))]
    mean = sum(s * p for s, p in zip(support, cur))
    ex2 = sum(s * s * p for s, p in zip(support, cur))
    out = {
        "support": support,
        "pmf": cur,
        "pmf_l": float("nan"),
        "sf_l": float("nan"),
        "mean": float(mean),
        "var": float(ex2 - mean * mean),
        "k": k,
        "n": n,
        "method": "exact null distribution of Page's L (Sec. 12.3)",
    }
    if ell is not None:
        li = int(round(float(ell))) - lo * k
        if 0 <= li < len(cur):
            out["pmf_l"] = cur[li]
            out["sf_l"] = float(sum(cur[li:]))
        else:
            out["pmf_l"] = 0.0
            out["sf_l"] = 0.0 if li >= len(cur) else 1.0
    return RichResult(payload=out)


gibbons_page_exact = pageexact
