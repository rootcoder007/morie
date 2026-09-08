# morie.fn -- function file (rootcoder007/morie)
"""Wilcoxon signed-rank statistic T+ and its null moments."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wsr', 'gibbons_wilcoxon_signed_rank']


def wsr(x, m0=0.0):
    """Signed-rank statistic T+ = sum of ranks of the positive |d|.

    Section 5.7 (book p. 195), eq. (5.7.1).  Zeros are dropped and N
    reduced (Sec. 5.7.1); |d| ties get midranks.  T- = N(N+1)/2 - T+.
    The reported z is eq. (5.7.9) with the tie correction (5.7.11)
    applied to the variance.

    Parameters
    ----------
    x : sequence of float
        Sample or paired differences.
    m0 : float, optional
        Hypothesised median (default 0).

    Returns
    -------
    RichResult
        keys ``statistic`` (T+), ``tminus``, ``n``, ``nzero``,
        ``mean``, ``var``, ``z``, ``p_value``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.7, eqs. (5.7.1), (5.7.2),
    (5.7.9), (5.7.11), pp. 195-203.
    """
    ds = [float(v) - float(m0) for v in x]
    nzero = sum(1 for v in ds if v == 0.0)
    ds = [v for v in ds if v != 0.0]
    n = len(ds)
    if n < 1:
        raise ValueError("no non-zero differences.")
    a = [abs(v) for v in ds]
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    corr = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        t = j - i + 1
        if t > 1:
            corr += t * (t * t - 1.0)
        i = j + 1
    tplus = sum(ranks[i] for i in range(n) if ds[i] > 0.0)
    mean = n * (n + 1.0) / 4.0
    var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - corr / 48.0
    z = (tplus - mean) / math.sqrt(var) if var > 0.0 else float("nan")
    pv = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if var > 0.0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(tplus),
            "tminus": float(n * (n + 1.0) / 2.0 - tplus),
            "n": int(n),
            "nzero": int(nzero),
            "mean": float(mean),
            "var": float(var),
            "z": float(z),
            "p_value": float(min(1.0, pv)),
            "method": "Wilcoxon signed-rank T+, eqs. (5.7.1)/(5.7.9)",
        }
    )


gibbons_wilcoxon_signed_rank = wsr
