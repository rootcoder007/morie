# morie.fn -- function file (rootcoder007/morie)
"""Best linear predictor for causal survival forest CATE."""

import numpy as np

from ._richresult import RichResult
from .crfhte import causal_forest_hte_test
from .csfgrf import causal_survival_forest

__all__ = ["causal_survival_blp"]


def causal_survival_blp(time, event, D, X, horizon=None, n_trees=200, min_leaf=15, seed=0):
    """Fit a causal survival forest, then test its CATE by the BLP.

    Chains :func:`morie.fn.csfgrf.causal_survival_forest` into
    :func:`morie.fn.crfhte.causal_forest_hte_test`, passing the
    *out-of-bag* CATE predictions and the IPCW-RMST pseudo outcome, so
    the heterogeneity coefficient is not inflated by in-bag reuse.

    Returns
    -------
    RichResult
        keys: ``alpha``, ``beta``, ``se_beta``, ``p_value``,
        ``heterogeneous``, ``ate``, ``horizon``, ``n``, ``method``.

    References
    ----------
    Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S. & Zhu, R. (2023).
    Estimating heterogeneous treatment effects with right-censored
    data via causal survival forests. *JRSS-B*, 85(2), 179-211.

    Chernozhukov, V., Demirer, M., Duflo, E. & Fernandez-Val, I.
    (2018). Generic machine learning inference on heterogenous
    treatment effects. arXiv:1712.04802.
    """
    f = causal_survival_forest(
        time, event, D, X, horizon=horizon, n_trees=n_trees, min_leaf=min_leaf, seed=seed
    )
    blp = causal_forest_hte_test(f["pseudo_outcome"], D, f["cate_oob"])
    return RichResult(
        payload={
            "alpha": blp["alpha"],
            "beta": blp["beta"],
            "se_beta": blp["se_beta"],
            "p_value": blp["p_value"],
            "heterogeneous": blp["heterogeneous"],
            "ate": f["ate"],
            "horizon": f["horizon"],
            "n": blp["n"],
            "method": "BLP calibration/heterogeneity test for a causal survival forest",
        }
    )


def cheatsheet():
    return "csurv2: csfgrf out-of-bag CATE into the BLP heterogeneity test"
