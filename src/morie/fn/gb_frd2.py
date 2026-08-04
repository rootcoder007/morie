# morie.fn -- function file (rootcoder007/morie)
"""Friedman statistic corrected for ties -- eq. (12.2.12)."""

import math

from ._richresult import RichResult

__all__ = ['friedties', 'gibbons_friedman_ties']


def friedties(data):
    """Tie correction for Friedman's Q, with the correction factor shown.

    Book p. 445, eq. (12.2.12).  Midranks within blocks reduce the
    denominator by the tie sum, so

    .. math:: Q = \\frac{12(n-1)S}{kn(n^2-1)
        - \\sum\\sum t(t^2-1)},

    the double sum over every tied set in every block.  Returning the
    uncorrected Q and the tie sum separately makes the size of the
    correction visible.

    Parameters
    ----------
    data : sequence of sequence of float
        k blocks of n treatment observations.

    Returns
    -------
    RichResult
        keys ``statistic`` (corrected Q), ``q_raw``, ``tiesum``,
        ``factor`` (corrected / raw), ``s``, ``rank_sums``, ``k``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (12.2.12), p. 445.
    """
    rows = [[float(v) for v in r] for r in data]
    k = len(rows)
    if k < 2:
        raise ValueError("need at least 2 blocks.")
    n = len(rows[0])
    if n < 2:
        raise ValueError("need at least 2 treatments.")
    rsum = [0.0] * n
    tiesum = 0.0
    for r in rows:
        if len(r) != n:
            raise ValueError("every block must have n observations.")
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
    s = sum((v - k * (n + 1.0) / 2.0) ** 2 for v in rsum)
    q0 = 12.0 * s / (k * n * (n + 1.0))
    den = k * n * (float(n) ** 2 - 1.0) - tiesum
    qc = 12.0 * (n - 1.0) * s / den if den > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(qc),
            "q_raw": float(q0),
            "tiesum": float(tiesum),
            "factor": float(qc / q0) if q0 != 0 else float("nan"),
            "s": float(s),
            "rank_sums": rsum,
            "k": int(k),
            "n": int(n),
            "method": "tie-corrected Friedman Q, eq. (12.2.12)",
        }
    )


gibbons_friedman_ties = friedties
