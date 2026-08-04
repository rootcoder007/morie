# morie.fn -- function file (rootcoder007/morie)
"""Risk difference between two proportions.

Rothman & Greenland, Modern Epidemiology: RD = R_1 - R_0, with the
binomial variance R(1-R)/n in each arm,

    Var(RD) = p_e (1 - p_e) / n_e + p_u (1 - p_u) / n_u.

Name collision, reported rather than papered over: ``risk_difference``
is also exported by :mod:`morie.fn.rd_es`, where it is a real
implementation taking the four cell counts (a, b, c, d) of a 2x2
table, and ``_lazy_map.json`` resolves the bare name there.  This
module keeps the proportion-based signature the stub declared, which
rd_es does not offer, and registers under its own compact alias.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["risk_difference"]

_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054,
      0.99: 2.5758293035489004}


def _zcrit(confidence):
    c = float(confidence)
    for k, v in _Z.items():
        if abs(c - k) < 1e-12:
            return v
    raise ValueError("confidence must be one of 0.90, 0.95, 0.99")


def risk_difference(p_exposed, p_unexposed, n_exposed=None,
                    n_unexposed=None, confidence=0.95):
    """Absolute difference in risk, R_1 - R_0.

    Parameters
    ----------
    p_exposed, p_unexposed : risks in the two arms, each in [0, 1].
    n_exposed, n_unexposed : optional arm sizes.  Supplied together
        they give the binomial standard error and a Wald interval.
    confidence : 0.90, 0.95 or 0.99.

    Returns
    -------
    RichResult with keys estimate, se, ci_lower, ci_upper,
    p_exposed, p_unexposed, confidence, method.

    References
    ----------
    Rothman & Greenland, Modern Epidemiology.
    """
    pe = float(p_exposed)
    pu = float(p_unexposed)
    for p in (pe, pu):
        if not 0.0 <= p <= 1.0:
            raise ValueError("risks must lie in [0, 1]")
    rd = pe - pu
    se = lo = hi = None
    if n_exposed is not None and n_unexposed is not None:
        ne = float(n_exposed)
        nu = float(n_unexposed)
        if ne <= 0 or nu <= 0:
            raise ValueError("arm sizes must be positive")
        se = math.sqrt(pe * (1 - pe) / ne + pu * (1 - pu) / nu)
        z = _zcrit(confidence)
        lo = rd - z * se
        hi = rd + z * se
    return with_describe_pointer(RichResult(payload={
        "estimate": float(rd), "se": se, "ci_lower": lo, "ci_upper": hi,
        "p_exposed": pe, "p_unexposed": pu,
        "confidence": float(confidence),
        "method": "risk difference (Rothman & Greenland)",
    }), "riskdf")


def cheatsheet():
    return "riskdf: Risk difference between two proportions"


# compact alias per ledger/NAMING.md
riskdiffp = risk_difference
