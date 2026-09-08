# morie.fn -- function file (rootcoder007/morie)
"""Friedman two-way analysis of variance by ranks."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['friedq', 'gibbons_friedman']


def friedq(data, correct=True):
    """Friedman's Q for a k x n table of blocks by treatments.

    Section 12.2 (book p. 441), eq. (12.2.8):

    .. math:: Q = \\frac{12 \\sum_{j=1}^{n} R_j^2}{kn(n+1)}
        - 3k(n+1),

    with k blocks (rows) and n treatments (columns) -- the book's
    orientation.  Ranks run within each block.  Ties handled by
    midranks use eq. (12.2.12),

    .. math:: Q = \\frac{12(n-1)S}{kn(n^2-1)
        - \\sum\\sum t(t^2-1)},
        \\qquad S = \\sum_j \\left[R_j
        - \\tfrac{k(n+1)}{2}\\right]^2,

    the double sum running over all tied sets in every block.  Q is
    asymptotically chi-square with n - 1 degrees of freedom.

    Parameters
    ----------
    data : sequence of sequence of float
        k rows (blocks) of n observations (treatments).
    correct : bool, optional
        Apply the tie correction of eq. (12.2.12) (default True).

    Returns
    -------
    RichResult
        keys ``statistic`` (Q), ``q_raw``, ``s``, ``df``, ``p_value``,
        ``rank_sums``, ``k`` (blocks), ``n`` (treatments), ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 12.2, eqs. (12.2.8) and
    (12.2.12), pp. 441-445 (Friedman, 1937, 1940).
    """
    rows = [[float(v) for v in r] for r in data]
    k = len(rows)
    if k < 2:
        raise ValueError("need at least 2 blocks.")
    n = len(rows[0])
    if n < 2:
        raise ValueError("need at least 2 treatments.")
    if any(len(r) != n for r in rows):
        raise ValueError("every block must have n observations.")
    rsum = [0.0] * n
    tiesum = 0.0
    for r in rows:
        order = sorted(range(n), key=lambda i: r[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and r[order[j + 1]] == r[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = mid
            tt = j - i + 1
            if tt > 1:
                tiesum += tt * (tt * tt - 1.0)
            i = j + 1
        for j in range(n):
            rsum[j] += rk[j]
    q = 12.0 / (k * n * (n + 1.0)) * sum(v * v for v in rsum) - 3.0 * k * (
        n + 1.0
    )
    s = sum((v - k * (n + 1.0) / 2.0) ** 2 for v in rsum)
    qc = q
    if correct and tiesum > 0.0:
        qc = 12.0 * (n - 1.0) * s / (
            k * n * (float(n) ** 2 - 1.0) - tiesum
        )
    return RichResult(
        payload={
            "statistic": float(qc),
            "q_raw": float(q),
            "s": float(s),
            "df": int(n - 1),
            "p_value": float(stats.chi2.sf(qc, n - 1)),
            "rank_sums": rsum,
            "k": int(k),
            "n": int(n),
            "method": "Friedman two-way ANOVA by ranks, eq. (12.2.8)",
        }
    )


gibbons_friedman = friedq
