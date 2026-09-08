# morie.fn -- k02 batch (rootcoder007/morie)
"""Higgins and Thompson's I^2 inconsistency statistic.

Source consulted: Higgins, J.P.T. and Thompson, S.G. (2002), Quantifying
heterogeneity in a meta-analysis, *Statistics in Medicine* 21, 1539-1558.
Equation (9) defines

    I^2 = 100 * (Q - (k - 1)) / Q,   truncated below at zero

and section 3.3 gives the confidence interval by transforming the interval
for H: ``I^2 = (H^2 - 1)/H^2``, with ``SE(ln H)`` from equations (7)-(8)

    Q > k:   ( ln Q - ln(k-1) ) / ( 2 (sqrt(2Q) - sqrt(2k - 3)) )
    Q <= k:  sqrt( (1/(2(k-2))) (1 - 1/(3 (k-2)^2)) )

Verified against ``metafor::rma`` (I2 = 15.0625788940795 on the fixture).
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02fe, k02pchi, k02z

from ._richresult import RichResult

__all__ = ["ma_higgins_i2"]


def ma_higgins_i2(yi, vi, level=0.95):
    """I^2 with the Higgins-Thompson confidence interval.

    Parameters
    ----------
    yi : array-like
        Study effect sizes.
    vi : array-like
        Within-study sampling variances.
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    RichResult
        estimate (I^2 in percent), se (of ln H), ci_lower, ci_upper, Q, df,
        p_Q, H, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    _mu, _var, _sw, q, df = k02fe(y, v)
    k = len(y)
    i2 = 100.0 * max(0.0, (q - df) / q)
    h = float(np.sqrt(q / df))
    if q > k:
        selnh = (float(np.log(q)) - float(np.log(df))) / (2.0 * (float(np.sqrt(2.0 * q)) - float(np.sqrt(2.0 * k - 3.0))))
    else:
        selnh = float(np.sqrt((1.0 / (2.0 * (k - 2))) * (1.0 - 1.0 / (3.0 * (k - 2) ** 2))))
    crit = k02z(0.5 + 0.5 * float(level))
    hlo = max(1.0, float(np.exp(float(np.log(h)) - crit * selnh)))
    hhi = float(np.exp(float(np.log(h)) + crit * selnh))
    return RichResult(
        payload={
            "estimate": float(i2),
            "se": float(selnh),
            "ci_lower": float(100.0 * (hlo**2 - 1.0) / hlo**2),
            "ci_upper": float(100.0 * (hhi**2 - 1.0) / hhi**2),
            "Q": float(q),
            "df": int(df),
            "p_Q": float(k02pchi(q, df)),
            "H": h,
            "n": int(k),
            "method": "I^2 inconsistency with Higgins-Thompson interval (Higgins & Thompson 2002, eq. 9)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_higgins_i2(y, v)
# >>> assert abs(r["estimate"] - 15.0625788940795) < 1e-11    # metafor rma I2
# >>> assert r["ci_lower"] <= r["estimate"] <= r["ci_upper"]


def cheatsheet():
    return "mai2(yi, vi): Higgins-Thompson I^2 with confidence interval."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
mahigginsi2 = ma_higgins_i2
