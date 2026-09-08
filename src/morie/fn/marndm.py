# morie.fn -- k02 batch (rootcoder007/morie)
"""DerSimonian-Laird random-effects meta-analysis.

Source consulted: DerSimonian, R. and Laird, N. (1986), Meta-analysis in
clinical trials, *Controlled Clinical Trials* 7, 177-188, equations (5)-(9).
The moment estimator of the between-study variance is

    tau^2 = max(0, (Q - (k - 1)) / C),   C = sum(w_i) - sum(w_i^2)/sum(w_i)

with ``w_i = 1/v_i`` the fixed-effect weights and ``Q`` Cochran's statistic.
The pooled effect then uses ``w*_i = 1/(v_i + tau^2)``.  Verified against
``metafor::rma(method="DL")`` (agreement to 1e-15).
"""

from __future__ import annotations

from . import _array_core as np
from .k02util import k02dl, k02p2z, k02pchi, k02z

from ._richresult import RichResult

__all__ = ["ma_random_dl"]


def ma_random_dl(yi, vi, level=0.95):
    """Random-effects pooled estimate by the DerSimonian-Laird method.

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
        estimate, se, ci_lower, ci_upper, z, p_value, tau2, tau, Q, df,
        p_Q, weights, n, method.
    """
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    tau2, mu, var, q, df = k02dl(y, v)
    se = float(np.sqrt(var))
    z = mu / se
    crit = k02z(0.5 + 0.5 * float(level))
    ws = 1.0 / (v + tau2)
    return RichResult(
        payload={
            "estimate": float(mu),
            "se": se,
            "ci_lower": float(mu - crit * se),
            "ci_upper": float(mu + crit * se),
            "z": float(z),
            "p_value": float(k02p2z(z)),
            "tau2": float(tau2),
            "tau": float(np.sqrt(tau2)),
            "Q": float(q),
            "df": int(df),
            "p_Q": float(k02pchi(q, df)),
            "weights": (ws / float(np.sum(ws))).tolist(),
            "n": int(len(y)),
            "method": "DerSimonian-Laird random-effects meta-analysis (DerSimonian & Laird 1986)",
        }
    )


# CANONICAL TEST
# >>> y = [0.10, 0.30, -0.20, 0.45, 0.05, 0.22]
# >>> v = [0.02, 0.05, 0.03, 0.08, 0.01, 0.04]
# >>> r = ma_random_dl(y, v)
# >>> assert abs(r["tau2"] - 0.00494218900675024) < 1e-14   # metafor rma DL
# >>> assert abs(r["estimate"] - 0.0920094772579361) < 1e-13
# >>> assert abs(r["se"] - 0.0729595735147854) < 1e-13


def cheatsheet():
    return "marndm(yi, vi): DerSimonian-Laird random-effects meta-analysis."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
marandomdl = ma_random_dl
