# morie.fn -- function file (rootcoder007/morie)
"""Loss-balanced (imbalance-regularized) causal forest -- GRF Sec 2.4."""

from __future__ import annotations

from . import _array_core as np

from ._cforest import CausalForest
from ._richresult import RichResult

__all__ = ["egregious_loss_forest"]


def egregious_loss_forest(y, D, X, n_trees=200, min_leaf=10, max_depth=6,
                          imbalance_penalty=100.0, subsample=0.5, seed=0):
    r"""Honest causal forest with GRF's split-imbalance regularizer.

    The heterogeneity criterion of Athey & Imbens rewards a split by

    .. math::
        \Delta = \frac{n_L n_R}{n}\,(\hat\tau_L - \hat\tau_R)^2,

    which is maximised, other things equal, by carving off the smallest
    admissible leaf: the extreme tau it reports is mostly estimation noise, and
    the noise itself is what earns the split. GRF prices that in by subtracting
    an imbalance term,

    .. math::
        \Delta_{pen} = \Delta - \gamma\left(\frac{1}{n_L} + \frac{1}{n_R}\right),

    which is large exactly when one child is tiny. The result is a forest whose
    leaf losses are balanced rather than concentrated -- hence the name.

    Estimation stays honest: each tree splits on one half of its subsample and
    fills its leaves from the untouched half, so a leaf's tau is not estimated
    from the data that chose it.

    Parameters
    ----------
    y : array-like
        Outcome ``(n,)``.
    D : array-like
        Binary treatment ``(n,)``, coded 0/1.
    X : array-like
        Covariates ``(n, p)``.
    n_trees, min_leaf, max_depth, subsample, seed
        Forest controls, passed through to the shared honest-forest core.
    imbalance_penalty : float
        :math:`\gamma` above. Zero recovers the unregularized criterion.

    Returns
    -------
    RichResult
        ``cate`` (out-of-bag per-unit effects), ``ate``, ``se``, ``ci``,
        ``leaf_sizes``, ``imbalance_penalty``.

    Notes
    -----
    ``se`` is the standard error of the mean of the out-of-bag CATEs. It
    is not the Wager-Athey infinitesimal-jackknife interval and does not
    account for the forest's own sampling variability, so treat it as a
    lower bound on the true uncertainty.

    References
    ----------
    Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random
        forests. *The Annals of Statistics*, 47(2), 1148-1178.
    Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous
        treatment effects using random forests. *JASA*, 113(523), 1228-1242.

    Examples
    --------
    Effect modification by the first covariate: treatment helps when
    ``X[:, 0] > 0`` and hurts otherwise. The forest recovers the sign split.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(800, 4))
    >>> D = (rng.random(800) < 0.5).astype(float)
    >>> tau = np.where(X[:, 0] > 0, 2.0, -2.0)
    >>> y = X[:, 1] + tau * D + rng.normal(0, 0.5, 800)
    >>> r = egregious_loss_forest(y, D, X, seed=1)
    >>> hi, lo = r["cate"][X[:, 0] > 0], r["cate"][X[:, 0] <= 0]
    >>> bool(np.nanmean(hi) > 1.0 and np.nanmean(lo) < -1.0)
    True

    The ATE averages the two halves to about zero, and its interval covers it.

    >>> bool(r["ci"][0] < 0 < r["ci"][1])
    True

    The penalty coarsens the partition: raising it yields fewer, larger
    leaves, monotonically.

    >>> sizes = [egregious_loss_forest(y, D, X, imbalance_penalty=g, seed=1)
    ...          for g in (0.0, 200.0, 1000.0)]
    >>> [s["n_leaves"] for s in sizes] == sorted(
    ...     [s["n_leaves"] for s in sizes], reverse=True)
    True
    >>> bool(np.mean(sizes[2]["leaf_sizes"]) > 2 * np.mean(sizes[0]["leaf_sizes"]))
    True

    Because gamma is in the criterion's own units, a value too small for the
    problem changes nothing at all -- here the effects are of size 2, so a
    gamma of 1 is invisible.

    >>> tiny = egregious_loss_forest(y, D, X, imbalance_penalty=1.0, seed=1)
    >>> bool(abs(tiny["n_leaves"] - sizes[0]["n_leaves"]) < 0.01 * sizes[0]["n_leaves"])
    True

    >>> egregious_loss_forest(y, D, X, imbalance_penalty=-1.0)
    Traceback (most recent call last):
        ...
    ValueError: imbalance_penalty must be non-negative, got -1.0.
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size}")

    forest = CausalForest(
        n_trees=n_trees, min_leaf=min_leaf, max_depth=max_depth,
        subsample=subsample, imbalance_penalty=imbalance_penalty, seed=seed,
    ).fit(X, y, D)
    cate = forest.predict(oob=True)

    good = np.isfinite(cate)
    k = int(good.sum())
    ate = float(np.mean(cate[good])) if k else float("nan")
    se = float(np.std(cate[good], ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
    ci = (ate - 1.96 * se, ate + 1.96 * se)

    sizes = []
    for tree in forest.trees_:
        stack = [tree]
        while stack:
            nd = stack.pop()
            if nd.feature is None:
                sizes.append(int(nd.n))
            else:
                stack.extend((nd.left, nd.right))

    warn = []
    if k < y.size:
        warn.append(f"{y.size - k} rows had no out-of-bag tree; their CATE is NaN")
    return RichResult(
        title="Loss-balanced causal forest",
        summary_lines=[("n", int(y.size)), ("trees", int(n_trees)),
                       ("ATE", ate), ("penalty", float(imbalance_penalty))],
        warnings=warn,
        payload={
            "cate": cate, "ate": ate, "se": se, "ci": ci,
            "leaf_sizes": sizes, "n_leaves": len(sizes),
            "imbalance_penalty": float(imbalance_penalty),
            "estimate": ate, "n": int(y.size),
            "method": "egregious_loss_forest",
        },
    )


def cheatsheet():
    return "egrgrf: honest causal forest + GRF imbalance penalty; stops tiny-leaf noise from winning splits"
