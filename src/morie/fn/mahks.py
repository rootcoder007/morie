# morie.fn -- k02 batch (rootcoder007/morie)
"""Hartung-Knapp variance adjustment for a random-effects meta-analysis.

Source consulted: Hartung, J. and Knapp, G. (2001), On tests of the overall
treatment effect in meta-analysis with normally distributed responses,
*Statistics in Medicine* 20, 1771-1782; and Sidik, K. and Jonkman, J.N.
(2002).  With random-effects weights ``w_i = 1/(v_i + tau^2)`` and pooled
effect ``mu``, the ordinary standard error is replaced by

    se_HK = sqrt( (1/(k-1)) sum w_i (y_i - mu)^2 / sum w_i )

and inference uses ``t`` on ``k - 1`` degrees of freedom rather than the
normal.  The multiplier is a scale estimate of the residual heterogeneity,
so it widens the interval exactly when the studies disagree more than the
weights say they should.  Verified against ``metafor::rma(test="knha")``:
se 0.0745588782641038, lower limit -0.0996502208931884 on the fixture.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02p2t, k02tq

from ._richresult import RichResult

__all__ = ["ma_hartung_knapp"]


def ma_hartung_knapp(yi, vi, level=0.95):
    """Random-effects pooling with the Hartung-Knapp standard error.

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
        estimate, se (Hartung-Knapp), se_dl, ci_lower, ci_upper, t, df,
        p_value, tau2, inflation, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    tau2, mu, var, _q, df = k02dl(y, v)
    w = 1.0 / (v + tau2)
    sw = float(np.sum(w))
    se_dl = float(np.sqrt(var))
    se = float(np.sqrt(float(np.sum(w * (y - mu) ** 2)) / ((k - 1) * sw)))
    tstat = mu / se
    crit = k02tq(0.5 + 0.5 * float(level), k - 1)
    return RichResult(
        payload={
            "estimate": float(mu),
            "se": se,
            "se_dl": se_dl,
            "ci_lower": float(mu - crit * se),
            "ci_upper": float(mu + crit * se),
            "t": float(tstat),
            "df": int(k - 1),
            "p_value": float(k02p2t(tstat, k - 1)),
            "tau2": float(tau2),
            "inflation": float(se / se_dl),
            "n": int(k),
            "method": "Hartung-Knapp adjusted random-effects meta-analysis (Hartung & Knapp 2001)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_hartung_knapp(y, v)
# >>> assert abs(r["se"] - 0.0745588782641038) < 1e-13     # metafor knha
# >>> assert abs(r["ci_lower"] + 0.0996502208931884) < 1e-13


def cheatsheet():
    return "mahks(yi, vi): Hartung-Knapp adjusted random-effects meta-analysis."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
mahartungknapp = ma_hartung_knapp
