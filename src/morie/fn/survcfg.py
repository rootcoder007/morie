# morie.fn -- function file (rootcoder007/morie)
"""Causal survival forest (grf-style front-end)."""

from ._richresult import RichResult
from .csfgrf import causal_survival_forest

__all__ = ["causal_survival_forest_grf"]


def causal_survival_forest_grf(time, event, D, X, horizon=None, n_trees=200, min_leaf=15, seed=0):
    """Front-end to :func:`morie.fn.csfgrf.causal_survival_forest`.

    Same IPCW-RMST honest forest (Cui et al. 2023, *JRSS-B* 85(2),
    179-211); kept as a separate entry point matching the grf naming.
    """
    out = causal_survival_forest(
        time, event, D, X, horizon=horizon, n_trees=n_trees, min_leaf=min_leaf, seed=seed
    )
    payload = dict(out)
    payload["method"] = "Causal survival forest (grf-style front-end)"
    return RichResult(payload=payload)


def cheatsheet():
    return "survcfg: front-end to csfgrf"
