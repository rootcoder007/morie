# morie.fn -- function file (rootcoder007/morie)
"""Percentile modified rank test for location -- eq. (8.3.5)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['pctrankloc', 'gibbons_pct_mod_rank_loc']


def pctrankloc(x, y, s=0.5, r=None):
    """Gastwirth's T_s - B_r location statistic.

    Section 8.3.3 (book p. 304), eq. (8.3.5).  With S = [Ns] + 1 and
    R = [Nr] + 1, only the upper S and lower R of the combined array
    are scored; everything in between gets a score of zero:

        N odd:   B_r = sum_{i<=R} (R - i + 1) Z_i,
                 T_s = sum_{i>N-S} (i - (N-S)) Z_i
        N even:  B_r = sum_{i<=R} (R - i + 1/2) Z_i,
                 T_s = sum_{i>N-S} (i - (N-S) - 1/2) Z_i

    T_s - B_r tests location (T_s + B_r tests scale, Ch. 9).  For N
    even with S = R the book gives E = 0 and

    .. math:: Var[T_s - B_r] = \\frac{mnS(4S^2-1)}{6N(N-1)}
        \\qquad (8.3.6),

    which is returned as ``var_book`` whenever it applies; ``var`` is
    always the general Theorem 7.3.2 value for the realised scores, so
    the two can be compared directly.

    Parameters
    ----------
    x, y : sequence of float
        The two samples; Z_i = 1 when the i-th smallest is an X.
    s : float, optional
        Upper percentile fraction (default 0.5).
    r : float, optional
        Lower percentile fraction (defaults to ``s``).

    Returns
    -------
    RichResult
        keys ``statistic`` (T_s - B_r), ``tupper``, ``blower``,
        ``var``, ``var_book``, ``z``, ``p_value``, ``S``, ``R``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 8.3.3, eqs. (8.3.5)-(8.3.6),
    pp. 304-305 (Gastwirth, 1965).
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
    tagged = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    tagged.sort(key=lambda p: (p[0], p[1]))
    z = [1 if t == 0 else 0 for _, t in tagged]
    S = int(math.floor(nn * s)) + 1
    R = int(math.floor(nn * r)) + 1
    S = min(S, nn)
    R = min(R, nn)
    half = 0.0 if nn % 2 else 0.5
    a = [0.0] * nn
    for i in range(1, R + 1):
        a[i - 1] -= R - i + 1.0 - half
    for i in range(nn - S + 1, nn + 1):
        a[i - 1] += i - (nn - S) - half
    blower = sum(
        (R - i + 1.0 - half) * z[i - 1] for i in range(1, R + 1)
    )
    tupper = sum(
        (i - (nn - S) - half) * z[i - 1] for i in range(nn - S + 1, nn + 1)
    )
    abar = sum(a) / nn
    ss = sum((v - abar) ** 2 for v in a)
    mean = m * abar
    var = m * n * ss / (nn * (nn - 1.0))
    vb = float("nan")
    if nn % 2 == 0 and S == R:
        vb = m * n * S * (4.0 * S * S - 1.0) / (6.0 * nn * (nn - 1.0))
    stat = tupper - blower
    zz = (stat - mean) / math.sqrt(var) if var > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(stat),
            "tupper": float(tupper),
            "blower": float(blower),
            "var": float(var),
            "var_book": float(vb),
            "z": float(zz),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(zz)))),
            "S": int(S),
            "R": int(R),
            "m": m,
            "n": n,
            "method": "percentile modified rank test for location (8.3.5)",
        }
    )


gibbons_pct_mod_rank_loc = pctrankloc
