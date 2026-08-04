# morie.fn -- k02 batch (rootcoder007/morie)
"""Fixed-effect (inverse-variance) meta-analysis.

Source consulted: Borenstein, Hedges, Higgins and Rothstein (2009),
*Introduction to Meta-Analysis*, chapter 11 ("Fixed-Effect Model"), and the
original derivation in Hedges and Olkin (1985).  Each study contributes its
effect ``y_i`` with within-study variance ``v_i``; the weights are
``w_i = 1/v_i`` and

    M = sum(w_i y_i) / sum(w_i),    Var(M) = 1 / sum(w_i)

Cochran's homogeneity statistic ``Q = sum(w_i (y_i - M)^2)`` is reported on
``k - 1`` degrees of freedom.  Verified against ``metafor::rma(method="FE")``
on the fixture in the canonical test below (agreement to 1e-15).
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02fe, k02p2z, k02pchi, k02z

from ._richresult import RichResult

__all__ = ["ma_fixed_effect"]


def ma_fixed_effect(yi, vi, level=0.95):
    """Fixed-effect inverse-variance pooled estimate.

    Parameters
    ----------
    yi : array-like
        Study effect sizes.
    vi : array-like
        Within-study sampling variances.
    level : float, default 0.95
        Confidence level for the two-sided interval.

    Returns
    -------
    RichResult
        estimate, se, ci_lower, ci_upper, z, p_value, Q, df, p_Q,
        weights, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    mu, var, sw, q, df = k02fe(y, v)
    se = float(np.sqrt(var))
    z = mu / se
    crit = k02z(0.5 + 0.5 * float(level))
    return RichResult(
        payload={
            "estimate": float(mu),
            "se": se,
            "ci_lower": float(mu - crit * se),
            "ci_upper": float(mu + crit * se),
            "z": float(z),
            "p_value": float(k02p2z(z)),
            "Q": float(q),
            "df": int(df),
            "p_Q": float(k02pchi(q, df)),
            "weights": (1.0 / v / sw).tolist(),
            "n": int(len(y)),
            "method": "Fixed-effect inverse-variance meta-analysis (Borenstein et al. 2009, ch. 11)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_fixed_effect(y, v)
# >>> assert abs(r["estimate"] - 0.0849480968858132) < 1e-13   # metafor rma FE
# >>> assert abs(r["se"] - 0.0644379479417843) < 1e-13
# >>> assert abs(r["Q"] - 5.88668685121107) < 1e-12


def cheatsheet():
    return "mafix(yi, vi): fixed-effect inverse-variance meta-analysis."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
mafixedeffect = ma_fixed_effect
