# morie.fn -- function file (rootcoder007/morie)
"""Spearman's coefficient of rank correlation R."""

import math

from ._richresult import RichResult

__all__ = ['spearrho', 'gibbons_spearman_rho']


def spearrho(x, y):
    """R from the sum of squared rank differences, eq. (11.3.2).

    Section 11.3 (book p. 407).  With midranks for ties,

    .. math:: R = 1 - \\frac{6\\sum_{i=1}^{n} D_i^2}{n(n^2-1)},
        \\qquad D_i = \\mathrm{rank}(X_i) - \\mathrm{rank}(Y_i).

    That shortcut is exact only when there are no ties, so when ties
    are present the Pearson correlation of the midranks -- Sec. 11.3.4,
    the tie-corrected form -- is returned as ``statistic`` and the
    shortcut value as ``r_shortcut``, with ``tied`` flagging which
    applies.

    Parameters
    ----------
    x, y : sequence of float
        Paired observations, n >= 3.

    Returns
    -------
    RichResult
        keys ``statistic`` (R), ``r_shortcut``, ``sumd2``, ``tied``,
        ``var`` (1/(n-1)), ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 11.3, eq. (11.3.2), p. 407;
    ties Sec. 11.3.4, p. 413.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    n = len(xs)
    if len(ys) != n:
        raise ValueError("x and y must have the same length.")
    if n < 3:
        raise ValueError("need at least 3 pairs.")

    def _rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = mid
            i = j + 1
        return r

    rx = _rank(xs)
    ry = _rank(ys)
    tied = int(len(set(xs)) < n or len(set(ys)) < n)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    short = 1.0 - 6.0 * d2 / (n * (float(n) ** 2 - 1.0))
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((v - mx) ** 2 for v in rx) * sum((v - my) ** 2 for v in ry)
    )
    full = num / den if den > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(full),
            "r_shortcut": float(short),
            "sumd2": float(d2),
            "tied": tied,
            "var": 1.0 / (n - 1.0),
            "n": n,
            "method": "Spearman rank correlation, eq. (11.3.2)",
        }
    )


gibbons_spearman_rho = spearrho
