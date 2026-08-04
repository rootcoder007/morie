# morie.fn -- function file (rootcoder007/morie)
"""Kendall's tau coefficient for partial correlation -- eq. (12.6.1)."""

import math

from ._richresult import RichResult

__all__ = ['taupartial', 'gibbons_rank_corr_partial']


def taupartial(x, y, z):
    """T_{XY.Z} from the 2 x 2 table of concordance agreements.

    Section 12.6 (book p. 467), eq. (12.6.1).  Over the m(m-1)/2 pairs,
    classify each by whether X agrees with Z and whether Y agrees with
    Z, giving X11, X12, X21, X22 as in Table 12.6.1; then

    .. math:: T_{XY.Z} = \\frac{X_{11}X_{22} - X_{12}X_{21}}
        {(X_{.1}X_{.2}X_{1.}X_{2.})^{1/2}},

    which lies in [-1, 1].  Pairs tied in any of the three variables
    contribute to neither concordance nor discordance and are counted
    separately as ``dropped``.

    Parameters
    ----------
    x, y, z : sequence of float
        Three rankings or measurements of the same m subjects, m >= 3.

    Returns
    -------
    RichResult
        keys ``statistic``, ``x11``, ``x12``, ``x21``, ``x22``,
        ``dropped``, ``npairs``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 12.6, eq. (12.6.1), Table
    12.6.1, p. 467.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    zs = [float(v) for v in z]
    n = len(xs)
    if len(ys) != n or len(zs) != n:
        raise ValueError("x, y and z must have the same length.")
    if n < 3:
        raise ValueError("need at least 3 subjects.")
    x11 = x12 = x21 = x22 = 0
    dropped = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (xs[j] > xs[i]) - (xs[j] < xs[i])
            sy = (ys[j] > ys[i]) - (ys[j] < ys[i])
            sz = (zs[j] > zs[i]) - (zs[j] < zs[i])
            if sx == 0 or sy == 0 or sz == 0:
                dropped += 1
                continue
            xc = sx * sz > 0
            yc = sy * sz > 0
            if yc and xc:
                x11 += 1
            elif yc and not xc:
                x12 += 1
            elif (not yc) and xc:
                x21 += 1
            else:
                x22 += 1
    c1 = x11 + x21
    c2 = x12 + x22
    r1 = x11 + x12
    r2 = x21 + x22
    den = c1 * c2 * r1 * r2
    stat = (
        (x11 * x22 - x12 * x21) / math.sqrt(den)
        if den > 0
        else float("nan")
    )
    return RichResult(
        payload={
            "statistic": float(stat),
            "x11": int(x11),
            "x12": int(x12),
            "x21": int(x21),
            "x22": int(x22),
            "dropped": int(dropped),
            "npairs": int(n * (n - 1) // 2),
            "n": n,
            "method": "Kendall partial tau T_XY.Z, eq. (12.6.1)",
        }
    )


gibbons_rank_corr_partial = taupartial
