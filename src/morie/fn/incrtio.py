# morie.fn -- function file (rootcoder007/morie)
"""Incidence rate ratio and its confidence interval.

Rothman & Greenland, Modern Epidemiology.  The text itself is not in
the corpus and the publisher blocks direct fetch, so the estimator and
its variance were taken from the OpenEpi "Comparing Two Person-Time
Rates" technical documentation, which states them in the form the
textbook uses and was read in full:

    IRR = IR_e / IR_u,     Var(ln IRR) = 1/a + 1/b,

with a the case count in the exposed and b the case count in the
unexposed.  The variance needs the counts, not the rates, so the
interval is only returned when the counts are supplied.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["incidence_rate_ratio"]

_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054,
      0.99: 2.5758293035489004}


def _zcrit(confidence):
    """Normal quantile at 1 - alpha/2.  Tabulated for the three
    conventional levels so both language arms use bit-identical
    constants rather than two different quantile routines."""
    c = float(confidence)
    for k, v in _Z.items():
        if abs(c - k) < 1e-12:
            return v
    raise ValueError("confidence must be one of 0.90, 0.95, 0.99")


def incidence_rate_ratio(IR_e, IR_u, cases_exposed=None,
                         cases_unexposed=None, confidence=0.95):
    """Ratio of the incidence rate in the exposed to the unexposed.

    Parameters
    ----------
    IR_e, IR_u : float, the two incidence rates.
    cases_exposed, cases_unexposed : optional case counts a and b.
        Supplied together they give Var(ln IRR) = 1/a + 1/b and a
        Wald interval on the log scale.
    confidence : 0.90, 0.95 or 0.99.

    Returns
    -------
    RichResult with keys estimate, ln_estimate, se_ln, ci_lower,
    ci_upper, confidence, method.  The interval entries are None when
    the counts are not supplied.

    References
    ----------
    Rothman & Greenland, Modern Epidemiology; OpenEpi PersonTime2
    technical documentation.
    """
    re_ = float(IR_e)
    ru = float(IR_u)
    if ru == 0:
        raise ValueError("unexposed incidence rate must be non-zero")
    irr = re_ / ru
    se = lo = hi = None
    if cases_exposed is not None and cases_unexposed is not None:
        a = float(cases_exposed)
        b = float(cases_unexposed)
        if a <= 0 or b <= 0:
            raise ValueError("case counts must be positive for a CI")
        se = math.sqrt(1.0 / a + 1.0 / b)
        z = _zcrit(confidence)
        lo = irr * math.exp(-z * se)
        hi = irr * math.exp(z * se)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(irr),
        "ln_estimate": float(math.log(irr)) if irr > 0 else float("-inf"),
        "se_ln": se, "ci_lower": lo, "ci_upper": hi,
        "confidence": float(confidence),
        "method": "incidence rate ratio (Rothman & Greenland)",
    }), "incrtio")


def cheatsheet():
    return "incrtio: Incidence rate ratio"


# compact alias per ledger/NAMING.md
irratio = incidence_rate_ratio
