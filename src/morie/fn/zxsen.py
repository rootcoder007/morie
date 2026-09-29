# morie.fn -- function file (rootcoder007/morie)
"""Spatial stacking ensemble"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_stacking(y, X, coords, *, n_blocks=3, k=5, seed=0, n_trees=50):
    r"""Spatial stacking ensemble

    Stacked regression (Wolpert 1992; Breiman 1996) of three learners on the
    spatial features: a linear trend-surface model, a random forest
    (:func:`morie.fn.zxsrf.spatial_rf`, ``n_trees``) and gradient boosting
    (:func:`morie.fn.zxsgb.spatial_gbm`). Out-of-fold predictions come from
    spatial block cross-validation (:func:`morie.fn.zxsbv.spatial_cv_block`
    folds); the stacking weights are their non-negative least-squares
    combination (Lawson-Hanson), as Breiman recommends, and the base learners
    are then refitted to all the data.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    n_blocks, k, seed : int
        Block-CV settings.
    n_trees : int
        Trees in the forest and boosting rounds.

    Returns
    -------
    DescriptiveResult
        ``value`` is the RMSE of the stacked out-of-fold predictions; ``extra`` has ``weights`` (ols, rf, gbm), ``fitted`` and the out-of-fold predictions ``oof``.

    References
    ----------
    Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.

    Breiman, L. (1996). Stacked regressions. *Machine Learning*, 24(1), 49-64.

    Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial,
    hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> r = spatial_stacking(y, X, S, n_blocks=2, k=2, n_trees=10)
    >>> [round(w, 10) for w in r.extra["weights"]]
    [0.999870668, 0.0, 0.0]
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, 1)
    n = len(yv)
    m = max(1, len(Z[0]) // 3)
    folds = sm.blocks(coords, int(n_blocks), int(k), seed)
    oof = [[0.0, 0.0, 0.0] for _ in range(n)]

    def learners(tr):
        Zt, yt = [Z[i] for i in tr], [yv[i] for i in tr]
        b = sm.ols_fit(Zt, yt)
        trees, _o = sm.forest(Zt, yt, int(n_trees), m, 3, None, seed)
        f0, gb, _F = sm.boost(Zt, yt, int(n_trees), 0.1, 2, 3)
        return lambda Q: [sm.ols_predict(b, Q), sm.forest_predict(trees, Q), sm.boost_predict(f0, gb, 0.1, Q)]

    for f in sorted(set(folds)):
        te = [i for i, g in enumerate(folds) if g == f]
        P = learners([i for i, g in enumerate(folds) if g != f])([Z[i] for i in te])
        for a, i in enumerate(te):
            oof[i] = [P[0][a], P[1][a], P[2][a]]
    w = sm.nnls(oof, yv)
    P = learners(list(range(n)))(Z)
    fit = [w[0] * P[0][i] + w[1] * P[1][i] + w[2] * P[2][i] for i in range(n)]
    stacked = [sm.math.fsum(a * b for a, b in zip(w, r)) for r in oof]
    return DescriptiveResult(
        name="zxsen", value=sm.rmse(yv, stacked), extra={"weights": w, "fitted": fit, "folds": folds, "oof": oof}
    )


spat = spatial_stacking


def cheatsheet() -> str:
    return "spatial_stacking(...) -> Spatial stacking ensemble"
