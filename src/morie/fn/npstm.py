# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric TMLE for a survival treatment effect (RMST)."""

import numpy as np

from ._richresult import RichResult
from ._tmle import tmle_ate
from .csfgrf import causal_survival_forest

__all__ = ["nonparametric_tmle_survival"]


def nonparametric_tmle_survival(time, event, A, W, horizon=None, trunc=0.01):
    r"""Targeted difference in restricted mean survival time.

    Uses the inverse-probability-of-censoring-weighted RMST pseudo
    outcome (built by :func:`morie.fn.csfgrf.causal_survival_forest`,
    whose Kaplan-Meier censoring estimate is nonparametric) and then
    targets the treatment contrast with the TMLE fluctuation. The
    result is a difference in expected months-alive-within-horizon,
    which unlike a hazard ratio is collapsible and stays interpretable
    when the proportional-hazards assumption fails.

    Parameters
    ----------
    time : array-like, shape (n,)
        Follow-up times.
    event : array-like of {0, 1}, shape (n,)
        1 = event, 0 = censored.
    A : array-like of {0, 1}, shape (n,)
        Treatment.
    W : array-like, shape (n, p) or (n,)
        Covariates.
    horizon : float, optional
        Restriction time; default the 90th percentile of ``time``.
    trunc : float, default 0.01
        Propensity truncation.

    Returns
    -------
    RichResult
        keys: ``rmst_difference``, ``se``, ``ci``, ``rmst1``,
        ``rmst0``, ``horizon``, ``n_events``, ``n``, ``method``.

    References
    ----------
    Moore, K. L. & van der Laan, M. J. (2009). Increasing power in
    randomized trials with right censored outcomes through covariate
    adjustment. *Journal of Biopharmaceutical Statistics*, 19(6),
    1099-1131.

    Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S. & Zhu, R. (2023).
    Estimating heterogeneous treatment effects with right-censored
    data via causal survival forests. *JRSS-B*, 85(2), 179-211.
    (the IPCW-RMST pseudo outcome)
    """
    f = causal_survival_forest(time, event, A, W, horizon=horizon, n_trees=1, min_leaf=5, seed=0)
    pseudo = f["pseudo_outcome"]
    out = tmle_ate(pseudo, A, W, trunc=trunc)
    return RichResult(
        payload={
            "rmst_difference": out["ate"],
            "se": out["se"],
            "ci": out["ci"],
            "rmst1": out["ey1"],
            "rmst0": out["ey0"],
            "horizon": f["horizon"],
            "n_events": int(np.asarray(event, dtype=float).sum()),
            "n": out["n"],
            "method": "TMLE on IPCW-RMST pseudo outcomes (difference in restricted mean survival)",
        }
    )


def cheatsheet():
    return "npstm: IPCW RMST pseudo outcome, then the TMLE ATE fluctuation"
