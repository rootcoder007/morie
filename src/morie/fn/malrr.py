# morie.fn -- k02 batch (rootcoder007/morie)
"""Log risk ratio and its sampling variance for a 2x2 table.

Source consulted: Borenstein, Hedges, Higgins and Rothstein (2009),
*Introduction to Meta-Analysis*, chapter 5, equations (5.1)-(5.2).  With
``n1 = a + b`` and ``n2 = c + d``,

    ln(RR) = ln( (a/n1) / (c/n2) ),   Var = 1/a - 1/n1 + 1/c - 1/n2

Zero-cell tables receive the ``add`` continuity correction on every cell.
Verified against ``metafor::escalc(measure="RR")``.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_log_risk_ratio"]


def ma_log_risk_ratio(ai, bi, ci, di, add=0.5, level=0.95):
    """Log risk ratio with the delta-method variance.

    Parameters
    ----------
    ai, bi : array-like
        Events and non-events in the treated arm.
    ci, di : array-like
        Events and non-events in the control arm.
    add : float, default 0.5
        Continuity correction added to every cell of a table with a zero.
    level : float, default 0.95
        Confidence level for the two-sided interval.

    Returns
    -------
    RichResult
        estimate (log RR), se, variance, risk_ratio, ci_lower, ci_upper,
        corrected, n, method.
    """
    a = np.atleast_1d(np.asarray(ai, dtype=float))
    b = np.atleast_1d(np.asarray(bi, dtype=float))
    c = np.atleast_1d(np.asarray(ci, dtype=float))
    d = np.atleast_1d(np.asarray(di, dtype=float))
    zero = (a == 0.0) | (b == 0.0) | (c == 0.0) | (d == 0.0)
    adj = np.where(zero, float(add), 0.0)
    a = a + adj
    b = b + adj
    c = c + adj
    d = d + adj
    n1 = a + b
    n2 = c + d
    yi = np.log((a / n1) / (c / n2))
    vi = 1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n2
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
            "risk_ratio": pick(np.exp(yi)),
            "ci_lower": pick(yi - crit * se),
            "ci_upper": pick(yi + crit * se),
            "corrected": int(np.sum(zero)),
            "n": int(len(yi)),
            "method": "Log risk ratio (Borenstein et al. 2009, eq. 5.1-5.2)",
        }
    )


# CANONICAL TEST
# >>> r = ma_log_risk_ratio(12, 38, 7, 43)
# >>> assert abs(r["estimate"] - 0.538996500732687) < 1e-13   # metafor escalc RR
# >>> assert abs(r["variance"] - 0.186190476190476) < 1e-13


def cheatsheet():
    return "malrr(ai, bi, ci, di): log risk ratio and its variance."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
malogriskratio = ma_log_risk_ratio
