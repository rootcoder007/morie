"""Pearson's chi-square test of normality."""

import math

from ._richresult import RichResult
from ._stats_core import chi2, norm

__all__ = ["pearson_normality_test"]


def pearson_normality_test(x, n_classes=None, adjust=True):
    r"""Pearson chi-square test of normality with equiprobable classes.

    Hedderich, Sachs & Reynarowych (2023, Sec. 7.2.4, the ``nortest``
    ``pearson.test`` call of p. 515): the data are assigned to
    :math:`k` classes equiprobable under :math:`N(\bar x, s^2)` -- class
    :math:`\lfloor 1 + k\Phi((x_i - \bar x)/s)\rfloor` -- with default
    :math:`k = \lceil 2n^{2/5}\rceil` (Moore 1986), and
    :math:`P = \sum (C_j - n/k)^2/(n/k)` is referred to :math:`\chi^2` on
    :math:`k - 3` degrees of freedom when ``adjust`` (the two estimated
    parameters) or :math:`k - 1` otherwise.

    Parameters
    ----------
    x : sequence of float
    n_classes : int, optional
    adjust : bool

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``n_classes``, ``df``.

    References
    ----------
    Moore, D. S. (1986). Tests of chi-squared type. In D'Agostino, R. B. &
    Stephens, M. A. (eds), Goodness-of-Fit Techniques. Marcel Dekker.
    Gross, J. & Ligges, U. (2015). nortest: Tests for normality. R package.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 5:
        raise ValueError("need at least 5 observations")
    k = math.ceil(2 * n**0.4) if n_classes is None else int(n_classes)
    m = sum(xs) / n
    s = math.sqrt(sum((v - m) ** 2 for v in xs) / (n - 1))
    counts = [0] * k
    for v in xs:
        c = int(math.floor(1 + k * float(norm.cdf((v - m) / s))))
        counts[min(max(c, 1), k) - 1] += 1
    e = n / k
    stat = sum((c - e) ** 2 / e for c in counts)
    df = k - (3 if adjust else 1)
    if df < 1:
        raise ValueError("too few classes for the degrees of freedom")
    return RichResult(
        title="Pearson chi-square normality test",
        summary_lines=[("P", stat), ("df", df)],
        payload={"statistic": stat, "p_value": float(chi2.sf(stat, df)), "n_classes": k, "df": df},
    )


def cheatsheet():
    return "pchnrm: Pearson chi-square normality test, equiprobable classes (nortest pearson.test)"
