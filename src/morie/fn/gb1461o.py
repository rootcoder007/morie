# morie.fn -- function file (rootcoder007/morie)
"""Linear rank test for ordered categorical data -- Section 14.6.1."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['linbylin', 'gibbons_ordered_categories']


def linbylin(table, scores=None):
    """T = sum_j w_j X_{1j} over ordered columns, Sec. 14.6.1.

    Book p. 531.  When the columns of a 2 x c table are ordered, the
    chi-square test of independence throws the ordering away; the book
    instead uses a linear rank statistic

    .. math:: T = \\sum_j w_j X_{1j},

    with increasing scores w_j -- the column midranks give exactly the
    Wilcoxon rank-sum test on the grouped data.  Under H0

    .. math:: E[T] = n_1 \\bar w, \\qquad
        Var[T] = \\frac{n_1 n_2}{N(N-1)}
            \\sum_j c_j (w_j - \\bar w)^2.

    The book's Example 14.6.2 -- the 2 x 3 table [[2, 3, 5],
    [4, 5, 1]] -- gives T = 126, mean 105, sd 12.44 and z = 1.688 with
    a one-sided asymptotic p-value of 0.0457.

    Parameters
    ----------
    table : sequence of sequence of float
        A 2 x c table with ordered columns.
    scores : sequence of float, optional
        Column scores w_j (defaults to the column midranks).

    Returns
    -------
    RichResult
        keys ``statistic`` (T), ``mean``, ``var``, ``sd``, ``z``,
        ``p_value`` (one-sided upper), ``p_twosided``, ``scores``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.6.1, p. 531, with
    Example 14.6.2, pp. 531-532.
    """
    tb = [[float(v) for v in row] for row in table]
    if len(tb) != 2:
        raise ValueError("table must have exactly 2 rows.")
    c = len(tb[0])
    if c < 2 or len(tb[1]) != c:
        raise ValueError("both rows must have the same length, >= 2.")
    cs = [tb[0][j] + tb[1][j] for j in range(c)]
    n1 = sum(tb[0])
    n2 = sum(tb[1])
    nn = n1 + n2
    if nn < 2:
        raise ValueError("the table must contain at least 2 observations.")
    if scores is None:
        w = []
        acc = 0.0
        for j in range(c):
            w.append(acc + (cs[j] + 1.0) / 2.0)
            acc += cs[j]
    else:
        w = [float(v) for v in scores]
        if len(w) != c:
            raise ValueError("scores must have length c.")
    t = sum(w[j] * tb[0][j] for j in range(c))
    wbar = sum(cs[j] * w[j] for j in range(c)) / nn
    mean = n1 * wbar
    var = (
        n1 * n2 / (nn * (nn - 1.0))
        * sum(cs[j] * (w[j] - wbar) ** 2 for j in range(c))
    )
    sd = math.sqrt(var) if var > 0 else float("nan")
    z = (t - mean) / sd if var > 0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(t),
            "mean": float(mean),
            "var": float(var),
            "sd": float(sd),
            "z": float(z),
            "p_value": float(1.0 - stats.norm.cdf(z)),
            "p_twosided": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "scores": w,
            "n": float(nn),
            "method": "linear rank test for ordered categories (Sec. 14.6.1)",
        }
    )


gibbons_ordered_categories = linbylin
