"""Pearson and adjusted residuals of a contingency table under independence."""

import math

from ._richresult import RichResult
from ._stats_core import chi2

__all__ = ["contingency_residuals"]


def contingency_residuals(table):
    r"""Pearson and adjusted (standardized) residuals for an r x c table.

    With :math:`\hat n_{ij} = n_{i.} n_{.j}/n`, :math:`\alpha_i = n_{i.}/n` and
    :math:`\beta_j = n_{.j}/n` (Hedderich, Sachs & Reynarowych 2023, eqs 7.357,
    7.359), the Pearson residual is :math:`(n_{ij} - \hat n_{ij})/\sqrt{\hat n_{ij}}`
    and the adjusted residual

    .. math::  \breve\epsilon_{ij} = \frac{n_{ij} - \hat n_{ij}}
               {\sqrt{\hat n_{ij}(1 - \alpha_i)(1 - \beta_j)}},

    approximately N(0, 1) under independence (``chisq.test``'s ``residuals``
    and ``stdres``).

    Parameters
    ----------
    table : r x c nested sequence of counts

    Returns
    -------
    RichResult
        ``expected``, ``pearson``, ``adjusted`` (lists of rows), ``statistic``,
        ``df``, ``p_value``.

    References
    ----------
    Haberman, S. J. (1973). The analysis of residuals in cross-classified
    tables. Biometrics 29, 205-220.
    """
    t = [[float(v) for v in row] for row in table]
    nr, nc = len(t), len(t[0])
    if nr < 2 or nc < 2 or any(len(row) != nc for row in t) or any(v < 0 for row in t for v in row):
        raise ValueError("need an r x c table (r, c >= 2) of non-negative counts")
    rs = [sum(row) for row in t]
    cs = [sum(t[i][j] for i in range(nr)) for j in range(nc)]
    n = sum(rs)
    if min(rs) == 0 or min(cs) == 0:
        raise ValueError("a margin is zero")
    e = [[rs[i] * cs[j] / n for j in range(nc)] for i in range(nr)]
    pr = [[(t[i][j] - e[i][j]) / math.sqrt(e[i][j]) for j in range(nc)] for i in range(nr)]
    ad = [
        [(t[i][j] - e[i][j]) / math.sqrt(e[i][j] * (1 - rs[i] / n) * (1 - cs[j] / n)) for j in range(nc)]
        for i in range(nr)
    ]
    x = sum(v * v for row in pr for v in row)
    df = (nr - 1) * (nc - 1)
    p = float(chi2.sf(x, df))
    return RichResult(
        title="Contingency table residuals",
        summary_lines=[("statistic", x), ("p_value", p)],
        payload={"expected": e, "pearson": pr, "adjusted": ad, "statistic": x, "df": df, "p_value": p},
    )


def cheatsheet():
    return "ctresid: (n_ij - e_ij)/sqrt(e_ij) and / sqrt(e_ij (1 - n_i./n)(1 - n_.j/n))"
