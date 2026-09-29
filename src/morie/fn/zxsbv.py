# morie.fn -- function file (rootcoder007/morie)
"""Spatial block cross-validation"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_cv_block(y, X, coords, *, n_blocks=3, k=5, seed=0, trend=1):
    r"""Spatial block cross-validation

    Spatial block cross-validation (Roberts et al. 2017): the bounding box is
    cut into an ``n_blocks x n_blocks`` grid, the occupied blocks are put in a
    random order (Philox uniforms, ``seed``) and dealt to ``k`` folds in turn,
    and each fold is predicted by the least-squares model on the spatial
    features ``[1, X, s1, s2]`` fitted to the other folds.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    n_blocks : int
        Blocks per axis.
    k : int
        Folds.
    seed : int
        Philox seed.
    trend : int
        Trend-surface degree.

    Returns
    -------
    DescriptiveResult
        ``value`` is the pooled CV RMSE; ``extra`` has ``folds``, ``fold_rmse`` and ``predictions``.

    References
    ----------
    Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial,
    hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> r = spatial_cv_block(y, X, S, n_blocks=2, k=2)
    >>> r.extra["folds"][:6], round(r.value, 12)
    ([0, 0, 0, 0, 0, 0], 0.164164551594)
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, trend)
    folds = sm.blocks(coords, int(n_blocks), int(k), seed)
    pred = [0.0] * len(yv)
    frm = []
    for f in sorted(set(folds)):
        te = [i for i, g in enumerate(folds) if g == f]
        tr = [i for i, g in enumerate(folds) if g != f]
        b = sm.ols_fit([Z[i] for i in tr], [yv[i] for i in tr])
        p = sm.ols_predict(b, [Z[i] for i in te])
        for i, v in zip(te, p):
            pred[i] = v
        frm.append(sm.rmse([yv[i] for i in te], p))
    return DescriptiveResult(
        name="zxsbv", value=sm.rmse(yv, pred), extra={"folds": folds, "fold_rmse": frm, "predictions": pred}
    )


spat = spatial_cv_block
spatialcvblock = spatial_cv_block


def cheatsheet() -> str:
    return "spatial_cv_block(...) -> Spatial block cross-validation"
