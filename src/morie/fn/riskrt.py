# morie.fn -- function file (rootcoder007/morie)
"""Risk ratio between two proportions.

Rothman & Greenland, Modern Epidemiology: RR = R_1 / R_0, with the
delta-method variance on the log scale,

    Var(ln RR) = (1 - p_e) / (n_e p_e) + (1 - p_u) / (n_u p_u),

which is the usual 1/a - 1/N_1 + 1/b - 1/N_0 rewritten in terms of the
risks.  The interval is symmetric on the log scale, never on the ratio
scale.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["risk_ratio"]

_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054,
      0.99: 2.5758293035489004}


def _zcrit(confidence):
    c = float(confidence)
    for k, v in _Z.items():
        if abs(c - k) < 1e-12:
            return v
    raise ValueError("confidence must be one of 0.90, 0.95, 0.99")


def risk_ratio(p_exposed, p_unexposed, n_exposed=None,
               n_unexposed=None, confidence=0.95):
    """Ratio of the risk in the exposed to the risk in the unexposed.

    Parameters
    ----------
    p_exposed, p_unexposed : risks in the two arms, each in [0, 1];
        the unexposed risk must be non-zero.
    n_exposed, n_unexposed : optional arm sizes, needed for the
        interval.
    confidence : 0.90, 0.95 or 0.99.

    Returns
    -------
    RichResult with keys estimate, ln_estimate, se_ln, ci_lower,
    ci_upper, p_exposed, p_unexposed, confidence, method.

    References
    ----------
    Rothman & Greenland, Modern Epidemiology.
    """
    pe = float(p_exposed)
    pu = float(p_unexposed)
    for p in (pe, pu):
        if not 0.0 <= p <= 1.0:
            raise ValueError("risks must lie in [0, 1]")
    if pu == 0:
        raise ValueError("unexposed risk must be non-zero")
    rr = pe / pu
    se = lo = hi = None
    if n_exposed is not None and n_unexposed is not None:
        ne = float(n_exposed)
        nu = float(n_unexposed)
        if ne <= 0 or nu <= 0:
            raise ValueError("arm sizes must be positive")
        if pe <= 0:
            raise ValueError("exposed risk must be positive for a CI")
        se = math.sqrt((1 - pe) / (ne * pe) + (1 - pu) / (nu * pu))
        z = _zcrit(confidence)
        lo = rr * math.exp(-z * se)
        hi = rr * math.exp(z * se)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(rr), "ln_estimate": float(math.log(rr)),
        "se_ln": se, "ci_lower": lo, "ci_upper": hi,
        "p_exposed": pe, "p_unexposed": pu,
        "confidence": float(confidence),
        "method": "risk ratio (Rothman & Greenland)",
    }), "riskrt")


def cheatsheet():
    return "riskrt: Risk ratio between two proportions"


# compact alias per ledger/NAMING.md
riskratio = risk_ratio
