# morie.fn -- function file (rootcoder007/morie)
"""Odds ratio for a 2 x 2 table by Woolf's logit method."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['oddsrat', 'gibbons_odds_ratio']


def oddsrat(table, alpha=0.05, cc=0.0):
    """Odds ratio, its Woolf logit interval and the associated test.

    For the 2 x 2 table [[a, b], [c, d]],

    .. math:: \\widehat{OR} = \\frac{ad}{bc}, \\qquad
        \\widehat{Var}[\\ln \\widehat{OR}]
        = \\frac1a + \\frac1b + \\frac1c + \\frac1d,

    giving the interval exp[ln OR +- z_{alpha/2} SE] and the test
    statistic chi^2 = (ln OR)^2 / Var on 1 degree of freedom.  Pass
    ``cc`` (conventionally 0.5) to add a constant to every cell when a
    zero would otherwise make the logit undefined.

    SOURCE NOTE: the odds ratio is NOT in Gibbons & Chakraborti
    (2011) -- the phrase does not occur anywhere in the book, whose
    2 x 2 association measures are the contingency coefficient
    (Sec. 14.2.1) and phi / Cramer's V.  The generated stub carried
    the Gibbons citation in error.  The method implemented here is
    therefore attributed to its actual primary source, Woolf (1955),
    which introduced the logit estimator and the reciprocal-cell
    variance used above.

    Parameters
    ----------
    table : sequence of sequence of float
        The 2 x 2 table [[a, b], [c, d]].
    alpha : float, optional
        Two-sided level (default 0.05).
    cc : float, optional
        Constant added to every cell (default 0.0).

    Returns
    -------
    RichResult
        keys ``estimate`` (OR), ``log_or``, ``se``, ``lower``,
        ``upper``, ``statistic`` (chi-square), ``df``, ``p_value``,
        ``method``.

    References
    ----------
    Woolf, B. (1955). On estimating the relation between blood group
    and disease. *Annals of Human Genetics*, 19(4), 251-253.
    """
    tb = [[float(v) + float(cc) for v in row] for row in table]
    if len(tb) != 2 or any(len(row) != 2 for row in tb):
        raise ValueError("table must be 2 x 2.")
    a, b = tb[0]
    c, d = tb[1]
    if min(a, b, c, d) <= 0.0:
        raise ValueError(
            "every cell must be positive for the logit method; "
            "pass cc=0.5 to add a continuity constant."
        )
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    orr = a * d / (b * c)
    lor = math.log(orr)
    var = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d
    se = math.sqrt(var)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    chi = lor * lor / var
    return RichResult(
        payload={
            "estimate": float(orr),
            "log_or": float(lor),
            "se": float(se),
            "lower": float(math.exp(lor - z * se)),
            "upper": float(math.exp(lor + z * se)),
            "statistic": float(chi),
            "df": 1,
            "p_value": float(stats.chi2.sf(chi, 1)),
            "method": "odds ratio, Woolf (1955) logit method",
        }
    )


gibbons_odds_ratio = oddsrat
