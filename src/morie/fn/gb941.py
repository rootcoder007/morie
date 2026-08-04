# morie.fn -- function file (rootcoder007/morie)
"""Siegel-Tukey scale test using interleaved rank assignment."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['sgltukey', 'gibbons_siegel_tukey']


def _tagged(xs, ys):
    """Pooled sample tagged 0 for X, 1 for Y, sorted with X first on ties."""
    t = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    t.sort(key=lambda p: (p[0], p[1]))
    return [1.0 if lab == 0 else 0.0 for _, lab in t]


def _lrmoments(a, m, n):
    """Theorem 7.3.2 moments of sum a_i Z_i under H0."""
    nn = m + n
    abar = sum(a) / nn
    ss = sum((v - abar) ** 2 for v in a)
    return m * abar, m * n * ss / (nn * (nn - 1.0))


def sgltukey(x, y):
    """Siegel-Tukey statistic: the first N integers, dealt from the ends.

    Section 9.4 (book p. 320).  The weights for N even are

        i : 1  2  3  4  5 ... N/2 ... N-3 N-2 N-1 N
        a : 1  4  5  8  9 ...  N  ...   7   6   3  2

    -- rank 1 to the smallest, 2 and 3 to the two largest, 4 and 5 to
    the next two smallest, and so on, so that sums of adjacent weight
    pairs are symmetric.  Because the weights are a permutation of
    1..N the null distribution is exactly that of the Wilcoxon
    rank-sum statistic, and the same tables serve.  When N is odd the
    middle observation is discarded and the weights are built for the
    reduced N.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``statistic``, ``mean``, ``var``, ``z``, ``p_value``,
        ``scores``, ``dropped`` (1 if the middle value was discarded),
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.4, p. 320
    (Siegel and Tukey, 1960).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    if len(xs) < 1 or len(ys) < 1:
        raise ValueError("both samples must be non-empty.")
    tag = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    tag.sort(key=lambda p: (p[0], p[1]))
    dropped = 0
    if len(tag) % 2 == 1:
        del tag[len(tag) // 2]
        dropped = 1
    nn = len(tag)
    m = sum(1 for _, lab in tag if lab == 0)
    n = nn - m
    if m < 1 or n < 1:
        raise ValueError("dropping the middle value emptied a sample.")
    a = [0.0] * nn
    lo, hi = 0, nn - 1
    nxt = 1
    while lo <= hi:
        a[lo] = float(nxt)
        nxt += 1
        if lo == hi:
            break
        a[hi] = float(nxt)
        nxt += 1
        hi -= 1
        if lo + 1 > hi:
            break
        a[hi] = float(nxt)
        nxt += 1
        hi -= 1
        lo += 1
        if lo > hi:
            break
        a[lo] = float(nxt)
        nxt += 1
        lo += 1
    stat = sum(a[i] for i in range(nn) if tag[i][1] == 0)
    mean, var = _lrmoments(a, m, n)
    zz = (stat - mean) / math.sqrt(var) if var > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(stat),
            "mean": float(mean),
            "var": float(var),
            "z": float(zz),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(zz)))),
            "scores": a,
            "dropped": int(dropped),
            "m": int(m),
            "n": int(n),
            "method": "Siegel-Tukey scale test (Sec. 9.4)",
        }
    )


gibbons_siegel_tukey = sgltukey
