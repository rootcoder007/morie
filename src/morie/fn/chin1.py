"""The (N - 1) chi-square test for a fourfold table."""

from ._richresult import RichResult
from ._stats_core import chi2

__all__ = ["chi_square_n_minus_1"]


def chi_square_n_minus_1(table):
    r"""Fourfold chi-square with :math:`n` replaced by :math:`n - 1`.

    Hedderich, Sachs & Reynarowych (2023, eq 7.277), recommended over the
    ordinary statistic for small n (Egon Pearson's 'N - 1' form):

    .. math::  \hat\chi^2_* = \frac{(n-1)(ad - bc)^2}{(a+b)(c+d)(a+c)(b+d)},

    on 1 df, with :math:`0 \le \hat\chi^2_* \le n - 1` (eq 7.278).

    Parameters
    ----------
    table : 2 x 2 nested sequence [[a, b], [c, d]]

    Returns
    -------
    RichResult
        ``statistic``, ``df``, ``p_value``.

    References
    ----------
    Campbell, I. (2007). Chi-squared and Fisher-Irwin tests of two-by-two
    tables with small sample recommendations. Statistics in Medicine 26,
    3661-3675.
    """
    (a, b), (c, d) = [[float(v) for v in row] for row in table]
    if min(a, b, c, d) < 0:
        raise ValueError("counts must be non-negative")
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        raise ValueError("a margin is zero")
    n = a + b + c + d
    x = (n - 1) * (a * d - b * c) ** 2 / den
    p = float(chi2.sf(x, 1))
    return RichResult(
        title="(N - 1) chi-square test",
        summary_lines=[("statistic", x), ("p_value", p)],
        payload={"statistic": x, "df": 1, "p_value": p},
    )


def cheatsheet():
    return "chin1: (n - 1)(ad - bc)^2 / ((a + b)(c + d)(a + c)(b + d)) on 1 df"
