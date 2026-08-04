# morie.fn -- k02 batch (rootcoder007/morie)
"""Log odds ratio and its sampling variance for a 2x2 table.

Source consulted: Borenstein, Hedges, Higgins and Rothstein (2009),
*Introduction to Meta-Analysis*, chapter 5, equations (5.8)-(5.9); the
variance is Woolf's (1955).  For a table with ``a`` events and ``b``
non-events in the treated arm and ``c``, ``d`` in the control arm,

    ln(OR) = ln( a d / (b c) ),   Var = 1/a + 1/b + 1/c + 1/d

A continuity correction ``add`` is applied to every cell of a table that
contains a zero (the ``metafor`` "only0" rule), which is what makes the
statistic finite for sparse tables.  Verified against
``metafor::escalc(measure="OR")``.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_log_odds_ratio"]


def ma_log_odds_ratio(ai, bi, ci, di, add=0.5, level=0.95):
    """Log odds ratio with Woolf's variance.

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
        estimate (log OR), se, variance, odds_ratio, ci_lower, ci_upper,
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
    yi = np.log(a * d / (b * c))
    vi = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d
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
            "odds_ratio": pick(np.exp(yi)),
            "ci_lower": pick(yi - crit * se),
            "ci_upper": pick(yi + crit * se),
            "corrected": int(np.sum(zero)),
            "n": int(len(yi)),
            "method": "Log odds ratio with Woolf variance (Borenstein et al. 2009, eq. 5.8-5.9)",
        }
    )


# CANONICAL TEST
# >>> r = ma_log_odds_ratio(12, 38, 7, 43)
# >>> assert abs(r["estimate"] - 0.662610456699864) < 1e-13   # metafor escalc OR
# >>> assert abs(r["variance"] - 0.275762079617649) < 1e-13


def cheatsheet():
    return "malor(ai, bi, ci, di): log odds ratio and Woolf variance."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
malogoddsratio = ma_log_odds_ratio
