# morie.fn -- function file (rootcoder007/morie)
"""Chi-square test of independence in an r x c contingency table."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['chiindep', 'gibbons_chisq_contingency']


def chiindep(table, correct=False):
    """Pearson Q for independence, Sec. 14.2.

    Book p. 505.  With f_ij the observed counts and
    e_ij = (row_i)(col_j)/N the expected counts under independence,

    .. math:: Q = \\sum_i\\sum_j \\frac{(f_{ij}-e_{ij})^2}{e_{ij}},

    asymptotically chi-square with (r-1)(c-1) degrees of freedom.
    Yates's continuity correction is available for the 2 x 2 case.

    Parameters
    ----------
    table : sequence of sequence of float
        The r x c table of counts, r, c >= 2.
    correct : bool, optional
        Apply Yates's correction (2 x 2 only, default False).

    Returns
    -------
    RichResult
        keys ``statistic``, ``df``, ``p_value``, ``expected``,
        ``n``, ``r``, ``c``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 14.2, p. 505.
    """
    tb = [[float(v) for v in row] for row in table]
    r = len(tb)
    if r < 2:
        raise ValueError("need at least 2 rows.")
    c = len(tb[0])
    if c < 2:
        raise ValueError("need at least 2 columns.")
    if any(len(row) != c for row in tb):
        raise ValueError("all rows must have the same length.")
    rs = [sum(row) for row in tb]
    cs = [sum(tb[i][j] for i in range(r)) for j in range(c)]
    nn = sum(rs)
    if nn <= 0:
        raise ValueError("the table must contain positive counts.")
    exp = [[rs[i] * cs[j] / nn for j in range(c)] for i in range(r)]
    yates = correct and r == 2 and c == 2
    q = 0.0
    for i in range(r):
        for j in range(c):
            e = exp[i][j]
            if e <= 0.0:
                raise ValueError("an expected frequency is zero.")
            d = abs(tb[i][j] - e)
            if yates:
                d = max(0.0, d - 0.5)
            q += d * d / e
    df = (r - 1) * (c - 1)
    return RichResult(
        payload={
            "statistic": float(q),
            "df": int(df),
            "p_value": float(stats.chi2.sf(q, df)),
            "expected": exp,
            "n": float(nn),
            "r": int(r),
            "c": int(c),
            "method": "chi-square test of independence (Sec. 14.2)",
        }
    )


gibbons_chisq_contingency = chiindep
