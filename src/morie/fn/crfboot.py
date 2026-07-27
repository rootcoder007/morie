# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap-honest causal forest."""

import numpy as np

from ._cforest import CausalForest
from ._richresult import RichResult

__all__ = ["causal_forest_bootstrap"]


def causal_forest_bootstrap(y, D, X, B=40, n_trees=60, min_leaf=10, seed=0, alpha=0.05):
    r"""Bootstrap confidence intervals for the forest CATE.

    Refits an honest causal forest on each of B bootstrap resamples
    and takes percentile intervals of :math:`\hat\tau(x)` across the
    replicates. Intervals from a *single* forest's tree spread
    understate the uncertainty because the trees share the training
    data; resampling the data is what makes the interval reflect
    sampling variability. (The bootstrap-of-little-bags in `grf` is
    cheaper; this is the direct version.)

    Parameters
    ----------
    y, D, X :
        As in :func:`morie.fn.crfath.causal_forest_wager_athey`.
    B : int, default 40
        Bootstrap replicates.
    n_trees, min_leaf, seed :
        Per-forest hyperparameters.
    alpha : float, default 0.05
        Two-sided interval level.

    Returns
    -------
    RichResult
        keys: ``cate`` (mean across replicates), ``ci_low``,
        ``ci_high`` (n,), ``se`` (n,), ``ate``, ``ate_ci``, ``B``,
        ``n``, ``method``.

    References
    ----------
    Wager, S. & Athey, S. (2018). Estimation and inference of
    heterogeneous treatment effects using random forests. *JASA*,
    113(523), 1228-1242.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    n = y.size
    B = int(B)
    if B < 2:
        raise ValueError(f"B must be at least 2, got {B}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    rng = np.random.default_rng(seed)
    draws = np.empty((B, n))
    for b in range(B):
        idx = rng.integers(0, n, n)
        f = CausalForest(n_trees=n_trees, min_leaf=min_leaf, seed=seed + b + 1)
        f.fit(X[idx], y[idx], D[idx])
        draws[b] = f.predict(X)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci = np.percentile(draws, [lo, hi], axis=0)
    ate_draws = draws.mean(axis=1)

    return RichResult(
        payload={
            "cate": draws.mean(axis=0),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "se": draws.std(axis=0, ddof=1),
            "ate": float(ate_draws.mean()),
            "ate_ci": (float(np.percentile(ate_draws, lo)), float(np.percentile(ate_draws, hi))),
            "B": B,
            "n": int(n),
            "method": "Bootstrap-honest causal forest (percentile CATE intervals)",
        }
    )


def cheatsheet():
    return "crfboot: refit the honest forest on B resamples; percentile CATE intervals"
