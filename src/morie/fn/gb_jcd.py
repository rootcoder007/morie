# morie.fn -- function file (rootcoder007/morie)
"""Jonckheere-Terpstra statistic as the matrix of pairwise counts."""

import math

from ._richresult import RichResult

__all__ = ['jtsum', 'gibbons_jt_cd_form']


def jtsum(samples):
    """The full U_ij matrix whose upper triangle sums to B.

    Section 10.6 (book p. 365) writes the alternative as the
    k(k-1)/2 pairwise statements of eq. (10.6.1), one per ordered pair,
    and B as the plain sum of the corresponding Mann-Whitney counts.
    Returning the whole matrix makes the decomposition visible: each
    U_ij can be inspected on its own, and their sum is B.

    Parameters
    ----------
    samples : sequence of sequence of float
        The k samples in the hypothesised order.

    Returns
    -------
    RichResult
        keys ``u`` (k x k matrix, upper triangle filled), ``statistic``
        (their sum), ``npairs``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.6, eq. (10.6.1), p. 365.
    """
    ss = [[float(v) for v in s] for s in samples]
    k = len(ss)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    u = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            c = 0.0
            for a in ss[i]:
                for b in ss[j]:
                    if a < b:
                        c += 1.0
                    elif a == b:
                        c += 0.5
            u[i][j] = c
    total = sum(u[i][j] for i in range(k) for j in range(i + 1, k))
    return RichResult(
        payload={
            "u": u,
            "statistic": float(total),
            "npairs": int(k * (k - 1) // 2),
            "k": int(k),
            "n": int(sum(len(s) for s in ss)),
            "method": "JT as the sum of pairwise Mann-Whitney counts",
        }
    )


gibbons_jt_cd_form = jtsum
