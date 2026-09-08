# morie.fn -- k02 batch (rootcoder007/morie)
"""Hartung-Knapp-Sidik-Jonkman interval and its t prediction interval.

Source consulted: IntHout, J., Ioannidis, J.P.A. and Borm, G.F. (2014), The
Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is
straightforward and considerably outperforms the standard DerSimonian-Laird
method, *BMC Medical Research Methodology* 14:25, section "Methods".  The
pooled effect is the usual random-effects estimate, the standard error is the
Hartung-Knapp quadratic form, and both the confidence interval and the
prediction interval use ``t`` on ``k - 1`` degrees of freedom:

    CI = mu +/- t_{k-1} se_HKSJ
    PI = mu +/- t_{k-1} sqrt(tau^2 + se_HKSJ^2)

Verified against ``metafor::rma(test="knha")`` and its ``predict()``:
CI lower -0.0996502208931884, PI (-0.171412021529504, 0.355430976045376).
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02p2t, k02tq

from ._richresult import RichResult

__all__ = ["ma_hksj_t_pi"]


def ma_hksj_t_pi(yi, vi, level=0.95):
    """HKSJ confidence interval and t prediction interval.

    Parameters
    ----------
    yi : array-like
        Study effect sizes.
    vi : array-like
        Within-study sampling variances.
    level : float, default 0.95
        Level for both the confidence and the prediction interval.

    Returns
    -------
    RichResult
        estimate, se, ci_lower, ci_upper, pi_lower, pi_upper, t, df,
        p_value, tau2, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    tau2, mu, _var, _q, _df = k02dl(y, v)
    w = 1.0 / (v + tau2)
    sw = float(np.sum(w))
    se = float(np.sqrt(float(np.sum(w * (y - mu) ** 2)) / ((k - 1) * sw)))
    tstat = mu / se
    crit = k02tq(0.5 + 0.5 * float(level), k - 1)
    spread = float(np.sqrt(tau2 + se * se))
    return RichResult(
        payload={
            "estimate": float(mu),
            "se": se,
            "ci_lower": float(mu - crit * se),
            "ci_upper": float(mu + crit * se),
            "pi_lower": float(mu - crit * spread),
            "pi_upper": float(mu + crit * spread),
            "pi_spread": spread,
            "t": float(tstat),
            "df": int(k - 1),
            "p_value": float(k02p2t(tstat, k - 1)),
            "tau2": float(tau2),
            "n": int(k),
            "method": "HKSJ interval with t prediction interval (IntHout, Ioannidis & Borm 2014)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_hksj_t_pi(y, v)
# >>> assert abs(r["ci_lower"] + 0.0996502208931884) < 1e-13   # metafor knha
# >>> assert abs(r["pi_lower"] + 0.171412021529504) < 1e-12    # metafor predict
# >>> assert abs(r["pi_upper"] - 0.355430976045376) < 1e-12


def cheatsheet():
    return "mahsj(yi, vi): HKSJ interval plus t prediction interval."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
mahksjtpi = ma_hksj_t_pi
