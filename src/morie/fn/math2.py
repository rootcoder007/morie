# morie.fn -- k02 batch (rootcoder007/morie)
"""Higgins and Thompson's H (and H^2) heterogeneity statistic.

Source consulted: Higgins, J.P.T. and Thompson, S.G. (2002), Quantifying
heterogeneity in a meta-analysis, *Statistics in Medicine* 21, 1539-1558,
equations (6)-(8).  ``H^2 = Q / (k - 1)`` is the ratio of Cochran's Q to its
degrees of freedom, so H = 1 exactly under homogeneity, and the interval for
ln H uses the same standard error as the I^2 interval:

    Q > k:   SE(ln H) = ( ln Q - ln(k-1) ) / ( 2 (sqrt(2Q) - sqrt(2k - 3)) )
    Q <= k:  SE(ln H) = sqrt( (1/(2(k-2))) (1 - 1/(3 (k-2)^2)) )

Verified against ``metafor::rma`` (H2 = 1.17733737024221 on the fixture).
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02fe, k02pchi, k02z

from ._richresult import RichResult

__all__ = ["ma_higgins_h2"]


def ma_higgins_h2(yi, vi, level=0.95):
    """H^2 with the Higgins-Thompson confidence interval.

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
        estimate (H^2), H, se (of ln H), ci_lower, ci_upper, Q, df, p_Q,
        n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    _mu, _var, _sw, q, df = k02fe(y, v)
    k = len(y)
    h2 = q / df
    h = float(np.sqrt(h2))
    if q > k:
        selnh = (float(np.log(q)) - float(np.log(df))) / (2.0 * (float(np.sqrt(2.0 * q)) - float(np.sqrt(2.0 * k - 3.0))))
    else:
        selnh = float(np.sqrt((1.0 / (2.0 * (k - 2))) * (1.0 - 1.0 / (3.0 * (k - 2) ** 2))))
    crit = k02z(0.5 + 0.5 * float(level))
    hlo = max(1.0, float(np.exp(float(np.log(h)) - crit * selnh)))
    hhi = float(np.exp(float(np.log(h)) + crit * selnh))
    return RichResult(
        payload={
            "estimate": float(h2),
            "H": h,
            "se": float(selnh),
            "ci_lower": float(hlo**2),
            "ci_upper": float(hhi**2),
            "Q": float(q),
            "df": int(df),
            "p_Q": float(k02pchi(q, df)),
            "n": int(k),
            "method": "H^2 heterogeneity ratio (Higgins & Thompson 2002, eq. 6-8)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_higgins_h2(y, v)
# >>> assert abs(r["estimate"] - 1.17733737024221) < 1e-12    # metafor rma H2
# >>> assert abs(r["H"] ** 2 - r["estimate"]) < 1e-15


def cheatsheet():
    return "math2(yi, vi): Higgins-Thompson H^2 with confidence interval."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
mahigginsh2 = ma_higgins_h2
