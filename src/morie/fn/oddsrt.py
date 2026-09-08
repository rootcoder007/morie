# morie.fn -- function file (rootcoder007/morie)
"""Odds ratio from a 2 x 2 table.

Source: Cornfield, J. (1951), "A method of estimating comparative rates
from clinical data.  Applications to cancer of the lung, breast, and
cervix", *Journal of the National Cancer Institute* 11(6):1269-1275.
Cornfield's paper is the origin of the retrospective (case-control)
argument that the cross-product ratio

    OR = (a d) / (b c)

estimates the prospective relative risk when the disease is rare.  The
point estimate below is exactly that cross-product ratio.

The standard error is NOT Cornfield's.  Cornfield (1951) constructs an
interval by an iterative non-central-hypergeometric argument that this
module does not implement.  What is returned is the log-scale variance
of Woolf, B. (1955), "On estimating the relation between blood group and
disease", *Annals of Human Genetics* 19(4):251-253,

    Var(log OR) = 1/a + 1/b + 1/c + 1/d

which was NOT read directly and is used here in its standard published
statement.  The interval is therefore labelled a Woolf interval, not a
Cornfield interval, so that no result of this module is attributed to a
derivation it did not come from.

Cell layout follows the ledger formula OR = (a/b) / (c/d):

           exposed   unexposed
    case      a          c
    control   b          d
"""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["odds_ratio"]


def odds_ratio(a, b, c, d, conf_level=0.95, correction=0.0):
    """Cross-product odds ratio for a 2 x 2 table, with a Woolf interval.

    Parameters
    ----------
    a, b, c, d : float
        Cell counts.  ``a`` exposed cases, ``b`` exposed controls,
        ``c`` unexposed cases, ``d`` unexposed controls.
    conf_level : float
        Two-sided confidence level for the Woolf interval.
    correction : float
        Constant added to every cell before the ratio is formed.  Pass
        ``0.5`` for the Haldane-Anscombe correction when a cell is zero.
        The default adds nothing, so a zero cell yields a non-finite
        odds ratio rather than a silently altered table.

    Returns
    -------
    RichResult
        ``estimate``, ``log_estimate``, ``se_log``, ``ci_lower``,
        ``ci_upper``, ``z``, ``p_value``, ``n``.
    """
    if not 0.0 < conf_level < 1.0:
        raise ValueError("conf_level must lie strictly between 0 and 1")
    aa = float(a) + float(correction)
    bb = float(b) + float(correction)
    cc = float(c) + float(correction)
    dd = float(d) + float(correction)
    for v in (aa, bb, cc, dd):
        if v < 0.0:
            raise ValueError("2 x 2 cell counts must be non-negative")
    n = float(a) + float(b) + float(c) + float(d)
    if bb == 0.0 or cc == 0.0:
        est = float("inf")
    else:
        est = (aa * dd) / (bb * cc)
    if aa == 0.0 or bb == 0.0 or cc == 0.0 or dd == 0.0:
        nan = float("nan")
        return RichResult(payload={
            "estimate": est, "log_estimate": nan, "se_log": nan,
            "ci_lower": nan, "ci_upper": nan, "z": nan, "p_value": nan,
            "n": n, "conf_level": float(conf_level),
            "method": "Cornfield (1951) cross-product odds ratio; "
                      "a zero cell leaves the Woolf variance undefined"})
    log_or = math.log(est)
    var = 1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd
    se = var ** 0.5
    zq = float(stats.norm.ppf(0.5 + 0.5 * float(conf_level)))
    lo = log_or - zq * se
    hi = log_or + zq * se
    z = log_or / se
    p = 2.0 * (1.0 - float(stats.norm.cdf(abs(z))))
    return RichResult(payload={
        "estimate": float(est), "log_estimate": float(log_or),
        "se_log": float(se), "ci_lower": float(math.exp(lo)),
        "ci_upper": float(math.exp(hi)), "z": float(z), "p_value": float(p),
        "n": float(n), "conf_level": float(conf_level),
        "method": "Cornfield (1951) cross-product odds ratio, "
                  "Woolf (1955) log-scale variance 1/a+1/b+1/c+1/d"})


def cheatsheet():
    return "oddsrt: Cornfield (1951) odds ratio with a Woolf (1955) interval"


# compact alias per ledger/NAMING.md
oddsratio = odds_ratio
