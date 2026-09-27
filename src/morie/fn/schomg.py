"""Scored chi-square test of homogeneity of k samples over ordered categories."""

from ._richresult import RichResult
from ._stats_core import chi2

__all__ = ["score_homogeneity_test"]


def score_homogeneity_test(table, scores):
    r"""Test equality of k distributions over ordered categories by their mean scores.

    Rows of ``table`` are categories with scores :math:`x`, columns are the k
    samples. With column totals :math:`T_j = \sum_i x_i n_{ij}`,
    :math:`T = \sum_j T_j` (Hedderich, Sachs & Reynarowych 2023, eq 7.362),

    .. math::  \hat\chi^2 = \frac{(n-1)\{\sum_j T_j^2/n_j - T^2/n\}}
               {\sum_i x_i^2 n_{i.} - T^2/n},

    on k - 1 df; it equals :math:`(n-1)R^2` of the one-way ANOVA of the
    scores.

    Parameters
    ----------
    table : r x k nested sequence of counts
    scores : sequence of r floats

    Returns
    -------
    RichResult
        ``statistic``, ``df``, ``p_value``.

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, eq (7.362).
    """
    t = [[float(v) for v in row] for row in table]
    x = [float(v) for v in scores]
    r, k = len(t), len(t[0])
    if r != len(x) or r < 2 or k < 2 or any(len(row) != k for row in t):
        raise ValueError("need an r x k table (r, k >= 2) and r scores")
    nj = [sum(t[i][j] for i in range(r)) for j in range(k)]
    tj = [sum(x[i] * t[i][j] for i in range(r)) for j in range(k)]
    n, tt = sum(nj), sum(tj)
    if min(nj) <= 0:
        raise ValueError("every sample needs observations")
    den = sum(x[i] ** 2 * sum(t[i]) for i in range(r)) - tt * tt / n
    if den <= 0:
        raise ValueError("scores do not vary")
    s = (n - 1) * (sum(tj[j] ** 2 / nj[j] for j in range(k)) - tt * tt / n) / den
    p = float(chi2.sf(s, k - 1))
    return RichResult(
        title="Scored homogeneity test",
        summary_lines=[("statistic", s), ("p_value", p)],
        payload={"statistic": s, "df": k - 1, "p_value": p},
    )


def cheatsheet():
    return "schomg: (n - 1)(sum T_j^2/n_j - T^2/n) / (sum x^2 n_i. - T^2/n) on k - 1 df"
