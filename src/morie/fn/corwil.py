"""Williams' test for two dependent correlations sharing a variable."""

import math

from ._richresult import RichResult
from ._stats_core import t as tdist

__all__ = ["williams_test"]


def williams_test(r12, r13, r23, n):
    r"""Williams' T2 for :math:`H_0: \rho_{12} = \rho_{13}` in one sample.

    Steiger (1980, eq 7), the form Hedderich, Sachs & Reynarowych (2023,
    eq 7.394) cite from Williams (1959):

    .. math::

        T_2 = (r_{12} - r_{13})\sqrt{\frac{(n-1)(1+r_{23})}
        {2\frac{n-1}{n-3}|R| + \bar r^2(1-r_{23})^3}},\qquad
        \bar r = \frac{r_{12}+r_{13}}{2},

    :math:`|R| = 1 - r_{12}^2 - r_{13}^2 - r_{23}^2 + 2r_{12}r_{13}r_{23}`, on
    :math:`n-3` degrees of freedom. The book prints the last term of the
    denominator as :math:`(r_{12}+r_{13})^2/2\,(1-r_{23})^3` and its example
    follows it (2.15); Steiger's :math:`\bar r^2 = (r_{12}+r_{13})^2/4`
    gives 2.169, which is ``psych::r.test``.

    Parameters
    ----------
    r12, r13 : float
        The two correlations with the common variable 1.
    r23 : float
        The correlation between variables 2 and 3.
    n : int

    Returns
    -------
    RichResult
        ``statistic``, ``df``, ``p_value`` (two-sided), ``det_R``.

    References
    ----------
    Williams, E. J. (1959). The comparison of regression variables. JRSS B
    21, 396-399. Steiger, J. H. (1980). Tests for comparing elements of a
    correlation matrix. Psychological Bulletin 87, 245-251.
    """
    r12, r13, r23, n = float(r12), float(r13), float(r23), int(n)
    if n < 5:
        raise ValueError("need n >= 5")
    det = 1 - r12 * r12 - r13 * r13 - r23 * r23 + 2 * r12 * r13 * r23
    rb = (r12 + r13) / 2
    a = 2 * (n - 1) / (n - 3) * det + rb * rb * (1 - r23) ** 3
    stat = (r12 - r13) * math.sqrt((n - 1) * (1 + r23) / a)
    df = n - 3
    return RichResult(
        title="Williams' test for dependent correlations",
        summary_lines=[("T2", stat), ("df", df)],
        payload={"statistic": stat, "df": df, "p_value": float(2 * tdist.sf(abs(stat), df)), "det_R": det},
    )


def cheatsheet():
    return "corwil: Williams/Steiger T2 for rho12 = rho13, one sample"
