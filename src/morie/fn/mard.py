# morie.fn -- k02 batch (rootcoder007/morie)
"""Risk difference and its sampling variance for a 2x2 table.

Source consulted: Borenstein, Hedges, Higgins and Rothstein (2009),
*Introduction to Meta-Analysis*, chapter 5, equations (5.5)-(5.6).  With
``n1 = a + b`` and ``n2 = c + d``,

    RD = a/n1 - c/n2,   Var = a b / n1^3 + c d / n2^3

which is the sum of the two binomial variances of the arm risks.  The risk
difference stays defined at zero cells, so no continuity correction is
applied.  Verified against ``metafor::escalc(measure="RD")``.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_risk_difference"]


def ma_risk_difference(ai, bi, ci, di, level=0.95):
    """Risk difference with the binomial variance.

    Parameters
    ----------
    ai, bi : array-like
        Events and non-events in the treated arm.
    ci, di : array-like
        Events and non-events in the control arm.
    level : float, default 0.95
        Confidence level for the two-sided interval.

    Returns
    -------
    RichResult
        estimate, se, variance, risk1, risk2, ci_lower, ci_upper, n, method.
    """
    a = np.atleast_1d(np.asarray(ai, dtype=float))
    b = np.atleast_1d(np.asarray(bi, dtype=float))
    c = np.atleast_1d(np.asarray(ci, dtype=float))
    d = np.atleast_1d(np.asarray(di, dtype=float))
    n1 = a + b
    n2 = c + d
    p1 = a / n1
    p2 = c / n2
    yi = p1 - p2
    vi = a * b / n1**3 + c * d / n2**3
    se = np.sqrt(vi)
    from .k02util import k02z

    crit = k02z(0.5 + 0.5 * float(level))
    single = len(yi) == 1
    pick = (lambda arr: float(arr[0])) if single else (lambda arr: arr.tolist())
    return RichResult(
        payload={
            "estimate": pick(yi),
            "se": pick(se),
            "variance": pick(vi),
            "risk1": pick(p1),
            "risk2": pick(p2),
            "ci_lower": pick(yi - crit * se),
            "ci_upper": pick(yi + crit * se),
            "n": int(len(yi)),
            "method": "Risk difference (Borenstein et al. 2009, eq. 5.5-5.6)",
        }
    )


# CANONICAL TEST
# >>> r = ma_risk_difference(12, 38, 7, 43)
# >>> assert abs(r["estimate"] - 0.1) < 1e-14        # metafor escalc RD
# >>> assert abs(r["variance"] - 0.006056) < 1e-15


def cheatsheet():
    return "mard(ai, bi, ci, di): risk difference and its variance."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
mariskdifference = ma_risk_difference
