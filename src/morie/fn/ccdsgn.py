# morie.fn -- function file (rootcoder007/morie)
"""Unmatched case-control odds ratio."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["case_control"]


def case_control(cases, controls, exposed=None, unexposed=None, conf=0.95):
    """
    Unmatched case-control OR

    Formula: OR = ad/bc from the 2x2

    a exposed cases, b unexposed cases, c exposed controls, d unexposed
    controls.  In a case-control design the risks themselves are not
    identified -- sampling is on the outcome -- but the odds ratio is,
    which is Cornfield's point.  The interval uses Woolf's standard
    error on the log scale, sqrt(1/a + 1/b + 1/c + 1/d).

    Parameters
    ----------
    cases : array-like
        Either (a, b) counts, or a 0/1 exposure indicator per case.
    controls : array-like
        Either (c, d) counts, or a 0/1 exposure indicator per control.
    exposed, unexposed : ignored
        Kept for the stub signature; the exposure split comes from the
        first two arguments.
    conf : float
        Confidence level.

    Returns
    -------
    result : dict
        Keys: estimate (OR), a, b, c, d, log_or, se_log, ci_low,
        ci_high, chisq, significant, n.

    References
    ----------
    Cornfield (1951), Proc. 2nd Berkeley Symp. 4:135-148.
    Woolf (1955), Ann. Human Genetics 19(4):251-253.
    """
    cs = core.vec(cases)
    ct = core.vec(controls)
    if not cs or not ct:
        raise ValueError("empty input: cases and controls are required")
    if len(cs) == 2 and all(v == int(v) and v >= 0 for v in cs):
        a, b = cs[0], cs[1]
    else:
        if any(v not in (0.0, 1.0) for v in cs):
            raise ValueError("cases must be counts (a, b) or 0/1 indicators")
        a = sum(cs)
        b = len(cs) - a
    if len(ct) == 2 and all(v == int(v) and v >= 0 for v in ct):
        c, d = ct[0], ct[1]
    else:
        if any(v not in (0.0, 1.0) for v in ct):
            raise ValueError("controls must be counts (c, d) or 0/1 indicators")
        c = sum(ct)
        d = len(ct) - c
    if not (0.0 < conf < 1.0):
        raise ValueError("conf must lie strictly in (0, 1)")
    n = a + b + c + d
    if n <= 0:
        raise ValueError("the 2x2 table is empty")
    if b * c == 0.0:
        orr = float("inf") if a * d > 0.0 else float("nan")
    else:
        orr = (a * d) / (b * c)
    if min(a, b, c, d) > 0.0:
        se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        lo = math.log(orr)
        z = core.qnorm(0.5 + conf / 2.0)
        ci_l = math.exp(lo - z * se)
        ci_h = math.exp(lo + z * se)
    else:
        se = float("nan")
        lo = float("nan")
        ci_l = float("nan")
        ci_h = float("nan")
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d
    if min(r1, r2, c1, c2) > 0.0:
        chi = n * (a * d - b * c) ** 2 / (r1 * r2 * c1 * c2)
    else:
        chi = float("nan")
    sig = 1 if (chi == chi and chi > 3.841458820694124) else 0
    return RichResult(payload={
        "estimate": orr,
        "a": a, "b": b, "c": c, "d": d,
        "log_or": lo,
        "se_log": se,
        "ci_low": ci_l,
        "ci_high": ci_h,
        "chisq": chi,
        "significant": sig,
        "n": n,
        "method": "unmatched case-control odds ratio",
    })


def cheatsheet():
    return "ccdsgn: unmatched case-control odds ratio"


# compact alias per ledger/NAMING.md
casecontrol = case_control
