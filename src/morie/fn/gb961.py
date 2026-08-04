# morie.fn -- function file (rootcoder007/morie)
"""Percentile modified rank test for scale -- Section 9.6."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['pctranksc', 'gibbons_pct_mod_rank_sc']


def pctranksc(x, y, s=0.5, r=None):
    """T_s + B_r, the percentile modified scale statistic.

    Section 9.6 (book p. 323).  Adding the two pieces of eq. (8.3.5)
    instead of subtracting them gives weights symmetric about the
    middle of the array, which is what a scale test needs; for N even
    with S = R = N/2 this is the David-Barton type of test.  The book's
    moments for N even and S = R are

    .. math:: E[T_s+B_r] = \\frac{mS^2}{N}, \\qquad
        Var[T_s+B_r] = \\frac{mnS(4NS^2 - N - 6S^3)}{6N^2(N-1)},

    both returned as ``mean_book`` and ``var_book`` alongside the
    general Theorem 7.3.2 values for the realised scores.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.
    s : float, optional
        Upper percentile fraction (default 0.5).
    r : float, optional
        Lower percentile fraction (defaults to ``s``).

    Returns
    -------
    RichResult
        keys ``statistic``, ``tupper``, ``blower``, ``mean``, ``var``,
        ``mean_book``, ``var_book``, ``z``, ``p_value``, ``S``, ``R``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 9.6, p. 323, with eq. (8.3.5),
    p. 304 (Gastwirth, 1965; Gibbons and Gastwirth, 1966).
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    s = float(s)
    r = s if r is None else float(r)
    if not 0.0 < s <= 1.0 or not 0.0 < r <= 1.0:
        raise ValueError("s and r must lie in (0, 1].")
    nn = m + n
    tag = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    tag.sort(key=lambda p: (p[0], p[1]))
    z = [1.0 if lab == 0 else 0.0 for _, lab in tag]
    S = min(int(math.floor(nn * s)) + 1, nn)
    R = min(int(math.floor(nn * r)) + 1, nn)
    half = 0.0 if nn % 2 else 0.5
    a = [0.0] * nn
    for i in range(1, R + 1):
        a[i - 1] += R - i + 1.0 - half
    for i in range(nn - S + 1, nn + 1):
        a[i - 1] += i - (nn - S) - half
    blower = sum((R - i + 1.0 - half) * z[i - 1] for i in range(1, R + 1))
    tupper = sum(
        (i - (nn - S) - half) * z[i - 1] for i in range(nn - S + 1, nn + 1)
    )
    abar = sum(a) / nn
    ss = sum((v - abar) ** 2 for v in a)
    mean = m * abar
    var = m * n * ss / (nn * (nn - 1.0))
    mb = vb = float("nan")
    if nn % 2 == 0 and S == R:
        mb = m * S * S / float(nn)
        vb = (
            m * n * S * (4.0 * nn * S * S - nn - 6.0 * S**3)
            / (6.0 * float(nn) ** 2 * (nn - 1.0))
        )
    stat = tupper + blower
    zz = (stat - mean) / math.sqrt(var) if var > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(stat),
            "tupper": float(tupper),
            "blower": float(blower),
            "mean": float(mean),
            "var": float(var),
            "mean_book": float(mb),
            "var_book": float(vb),
            "z": float(zz),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(zz)))),
            "S": int(S),
            "R": int(R),
            "m": m,
            "n": n,
            "method": "percentile modified rank test for scale (Sec. 9.6)",
        }
    )


gibbons_pct_mod_rank_sc = pctranksc
