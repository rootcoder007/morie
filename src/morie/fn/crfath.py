# morie.fn -- function file (rootcoder007/morie)
"""Causal forest (Wager-Athey) for heterogeneous treatment effects."""

import numpy as np

from ._cforest import CausalForest
from ._richresult import RichResult

__all__ = ["causal_forest_wager_athey"]


def causal_forest_wager_athey(y, D, X, n_trees=200, min_leaf=10, max_depth=6, seed=0):
    r"""Honest causal forest CATE estimates.

    Fits :class:`morie.fn._cforest.CausalForest` and returns both
    in-bag and out-of-bag :math:`\hat\tau(x)` predictions plus the
    forest-average effect. Use the out-of-bag predictions for any
    downstream heterogeneity test -- in-bag values are contaminated by
    the trees that saw the row.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    n_trees, min_leaf, max_depth, seed :
        Forest hyperparameters.

    Returns
    -------
    RichResult
        keys: ``cate`` (n,), ``cate_oob`` (n,), ``ate`` (mean CATE),
        ``cate_sd``, ``n_trees``, ``n``, ``forest`` (the fitted
        object, for prediction at new X), ``method``.

    References
    ----------
    Wager, S. & Athey, S. (2018). Estimation and inference of
    heterogeneous treatment effects using random forests. *JASA*,
    113(523), 1228-1242.
    """
    f = CausalForest(n_trees=n_trees, min_leaf=min_leaf, max_depth=max_depth, seed=seed)
    f.fit(X, y, D)
    cate = f.predict()
    oob = f.predict(oob=True)
    return RichResult(
        payload={
            "cate": cate,
            "cate_oob": oob,
            "ate": float(np.nanmean(cate)),
            "cate_sd": float(np.nanstd(cate, ddof=1)),
            "n_trees": int(n_trees),
            "n": int(cate.size),
            "forest": f,
            "method": "Honest causal forest CATE (Wager-Athey)",
        }
    )


def cheatsheet():
    return "crfath: honest causal forest; use cate_oob for downstream tests"
