# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Doubly robust DiD with a treatment x covariate interaction.

The ATT is estimated separately at each level of the effect modifier V
with the doubly robust panel estimator of Sant'Anna and Zhao (2020),
"Doubly robust difference-in-differences estimators", Journal of
Econometrics 219(1):101-122, doi:10.1016/j.jeconom.2020.06.003,
equation (2.6):

    tau = E[(w1(D) - w0(D, X; pi)) (dY - mu_0(X))].

The contrast between the extreme levels of V is the interaction, the
DiD analogue of the effect modification discussed by Hernan and Robins
(2020), *Causal Inference: What If*, Chapman & Hall/CRC, chapter 13.
Because the level-specific influence functions are computed on disjoint
subsamples they are independent, so the variance of the contrast is the
sum of the level variances.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["interaction_did"]


def interaction_did(y, D, V, X):
    """Level-specific DR-DiD ATTs and their contrast.

    Parameters
    ----------
    y : array-like
        Outcome change dY (post minus pre) for each unit.
    D : array-like
        Treatment indicator, 0 or 1.
    V : array-like
        Effect modifier; its distinct values define the strata.
    X : array-like or None
        Covariates for the propensity and outcome models.
    """
    dy = core.vec(y)
    n = len(dy)
    if n == 0:
        raise ValueError("interaction_did: y is empty")
    d = core.vec(D)
    v = core.vec(V)
    if len(d) != n or len(v) != n:
        raise ValueError("interaction_did: y, D and V have different lengths")
    for t in d:
        if t not in (0.0, 1.0):
            raise ValueError("interaction_did: D must be 0 or 1")
    rows = core.mat(X) if X is not None else None
    if rows is not None and len(rows) != n:
        raise ValueError("interaction_did: X and y have different lengths")
    levels = []
    for t in v:
        if t not in levels:
            levels.append(t)
    levels.sort()
    atts = []
    ses = []
    counts = []
    for lv in levels:
        idx = [i for i in range(n) if v[i] == lv]
        sub_d = [d[i] for i in idx]
        if sum(sub_d) == 0 or sum(sub_d) == len(idx):
            raise ValueError("interaction_did: a level of V has only one treatment arm")
        sub_x = [rows[i] for i in idx] if rows is not None else None
        r = core.drdid_panel([dy[i] for i in idx], sub_d, sub_x)
        atts.append(r["tau"])
        ses.append(r["se"])
        counts.append(len(idx))
    if len(levels) >= 2:
        est = atts[-1] - atts[0]
        se = math.sqrt(ses[-1] ** 2 + ses[0] ** 2)
    else:
        est = atts[0]
        se = ses[0]
    full = core.drdid_panel(dy, d, rows)
    return RichResult(
        title="DR-DiD with treatment x covariate interaction",
        summary_lines=[("levels", len(levels)), ("interaction", est), ("se", se)],
        payload={
            "estimate": est,
            "se": se,
            "att": atts,
            "att_se": ses,
            "levels": levels,
            "level_n": counts,
            "att_overall": full["tau"],
            "se_overall": full["se"],
            "n_levels": len(levels),
            "n": n,
            "method": "DR-DiD (Sant'Anna & Zhao 2020 eq. 2.6) within levels of V; contrast = interaction",
        },
    )


def cheatsheet():
    return "itrct1: DR-DiD with treatment x covariate interaction"


# compact alias per ledger/NAMING.md
interactiondid = interaction_did
