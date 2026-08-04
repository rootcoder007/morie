# morie.fn -- k02 batch (rootcoder007/morie)
"""Two-step DerSimonian-Laird tau^2, started from the Hedges estimator.

Source consulted: DerSimonian, R. and Kacker, R. (2007), Random-effects model
for meta-analysis of clinical trials: an update, *Contemporary Clinical
Trials* 28, 105-114, equations (5)-(6).  Write ``a_i = 1/(v_i + tau0)`` for
arbitrary working weights; the generalised method-of-moments estimator is

    tau^2 = [ sum a_i (y_i - ybar_a)^2 - sum a_i v_i + (sum a_i^2 v_i)/sum a_i ]
            / [ sum a_i - (sum a_i^2)/sum a_i ]

With ``tau0 = 0`` (a_i = 1/v_i) this collapses algebraically to
DerSimonian-Laird, ``(Q - (k-1))/C`` -- the identity is asserted in the
canonical test.  The two-step estimator starts instead from the Hedges
(variance-components) estimator, which uses equal weights,

    tau0 = sum (y_i - ybar)^2 / (k - 1) - (1/k) sum v_i

verified against ``metafor::rma(method="HE")`` (0.0121333333333333 on the
fixture), and applies the display above once.
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02mm, k02p2z, k02z

from ._richresult import RichResult

__all__ = ["ma_two_step_dl_he"]


def ma_two_step_dl_he(yi, vi, level=0.95):
    """Two-step DerSimonian-Laird tau^2 with the Hedges start value.

    Parameters
    ----------
    yi : array-like
        Study effect sizes.
    vi : array-like
        Within-study sampling variances.
    level : float, default 0.95
        Confidence level for the two-sided interval on the pooled effect.

    Returns
    -------
    RichResult
        estimate (tau^2 after one step), tau2_he, tau2_dl, mu, se, ci_lower,
        ci_upper, z, p_value, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    k = len(y)
    ybar = float(np.mean(y))
    tau_he = max(0.0, float(np.sum((y - ybar) ** 2)) / (k - 1) - float(np.mean(v)))
    tau2 = k02mm(y, v, tau_he)
    tau_dl, _mu_dl, _var_dl, _q, _df = k02dl(y, v)
    ws = 1.0 / (v + tau2)
    sws = float(np.sum(ws))
    mu = float(np.sum(ws * y)) / sws
    se = float(np.sqrt(1.0 / sws))
    z = mu / se
    crit = k02z(0.5 + 0.5 * float(level))
    return RichResult(
        payload={
            "estimate": float(tau2),
            "tau2_he": float(tau_he),
            "tau2_dl": float(tau_dl),
            "mu": float(mu),
            "se": se,
            "ci_lower": float(mu - crit * se),
            "ci_upper": float(mu + crit * se),
            "z": float(z),
            "p_value": float(k02p2z(z)),
            "n": int(k),
            "method": "Two-step DerSimonian-Laird tau^2 from the Hedges start (DerSimonian & Kacker 2007)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_two_step_dl_he(y, v)
# >>> assert abs(r["tau2_he"] - 0.0121333333333333) < 1e-14   # metafor rma HE
# >>> assert abs(r["tau2_dl"] - 0.00494218900675024) < 1e-14  # metafor rma DL
# >>> # the general moment estimator at tau0 = 0 IS DerSimonian-Laird
# >>> from .k02util import k02mm
# >>> assert abs(k02mm(y, v, 0.0) - r["tau2_dl"]) < 1e-15


def cheatsheet():
    return "matr(yi, vi): two-step DerSimonian-Laird tau^2 from the Hedges start."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
matwostepdlhe = ma_two_step_dl_he
