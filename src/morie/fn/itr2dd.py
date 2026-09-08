# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Optimal individualized treatment regime chosen by DR-DiD value.

Athey and Imbens (2017), "The state of applied econometrics: causality
and policy evaluation", Journal of Economic Perspectives 31(2):3-32,
doi:10.1257/jep.31.2.3, section on policy evaluation: an individualized
rule is chosen by maximising the estimated value of the rule over a
restricted class.  Here the class is the threshold rules
d_jc(W) = 1{W_j > c} on the supplied covariates, and the value of a rule
is the doubly robust DiD ATT among the units the rule targets,

    d* = argmax_{j, c} tau_hat( {i : W_ij > c} ),

with tau_hat the panel estimator of Sant'Anna and Zhao (2020), Journal
of Econometrics 219(1):101-122, doi:10.1016/j.jeconom.2020.06.003,
equation (2.6).  The candidate thresholds are the deciles of each
covariate, so the search is finite and deterministic; subgroups without
both treatment arms, or smaller than min_frac of the sample, are
skipped.  No sample splitting is done, so the reported maximum is
optimistic -- that is a property of the method, not of this code, and
the ATT of the complementary group is reported alongside it.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["itr_optimal_did"]

_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def itr_optimal_did(y, D, W, min_frac=0.25):
    """Best threshold rule on W by doubly robust DiD value.

    Parameters
    ----------
    y : array-like
        Outcome change dY.
    D : array-like
        Treatment indicator, 0 or 1.
    W : array-like
        Covariates the rule may use, one row per unit.
    min_frac : float
        Smallest admissible targeted share of the sample.
    """
    dy = core.vec(y)
    n = len(dy)
    if n == 0:
        raise ValueError("itr_optimal_did: y is empty")
    d = core.vec(D)
    if len(d) != n:
        raise ValueError("itr_optimal_did: y and D have different lengths")
    for t in d:
        if t not in (0.0, 1.0):
            raise ValueError("itr_optimal_did: D must be 0 or 1")
    rows = core.mat(W)
    if len(rows) != n:
        raise ValueError("itr_optimal_did: W and y have different lengths")
    p = len(rows[0])
    if not (0.0 < min_frac < 1.0):
        raise ValueError("itr_optimal_did: min_frac must lie in (0, 1)")
    full = core.drdid_panel(dy, d, rows)
    best = None
    tried = 0
    for j in range(p):
        col = [rows[i][j] for i in range(n)]
        cand = []
        for q in _LEVELS:
            c = core.quantile7(col, q)
            if c not in cand:
                cand.append(c)
        for c in cand:
            idx = [i for i in range(n) if col[i] > c]
            if len(idx) < min_frac * n or len(idx) > (1.0 - min_frac) * n:
                continue
            sub_d = [d[i] for i in idx]
            if sum(sub_d) == 0 or sum(sub_d) == len(idx):
                continue
            r = core.drdid_panel([dy[i] for i in idx], sub_d, [rows[i] for i in idx])
            tried += 1
            if best is None or r["tau"] > best[0]:
                best = (r["tau"], j, c, len(idx), r["se"])
    if best is None:
        raise ValueError("itr_optimal_did: no admissible threshold rule")
    tau, j, c, m, se = best
    comp = [i for i in range(n) if rows[i][j] <= c]
    cd = [d[i] for i in comp]
    if 0 < sum(cd) < len(comp):
        rc = core.drdid_panel([dy[i] for i in comp], cd, [rows[i] for i in comp])
        tau_c = rc["tau"]
    else:
        tau_c = float("nan")
    return RichResult(
        title="Optimal individualized treatment regime via DR-DiD",
        summary_lines=[("rule", "W%d > %.6g" % (j, c)), ("value", tau), ("targeted", m)],
        payload={
            "estimate": tau,
            "se": se,
            "feature": float(j),
            "threshold": c,
            "n_targeted": m,
            "share_targeted": m / float(n),
            "att_complement": tau_c,
            "att_overall": full["tau"],
            "se_overall": full["se"],
            "gain": tau - full["tau"],
            "n_rules": tried,
            "n": n,
            "method": "d*(W) = argmax_{j,c} DR-DiD ATT on {W_j > c}, Athey & Imbens (2017)",
        },
    )


def cheatsheet():
    return "itr2dd: Optimal individualized treatment regime via DR-DiD"


# compact alias per ledger/NAMING.md
itroptimaldid = itr_optimal_did
