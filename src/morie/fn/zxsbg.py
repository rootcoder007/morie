# morie.fn -- function file (rootcoder007/morie)
"""Spatial bagging"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_bagging(y, X, coords, *, n_trees=100, min_leaf=5, max_depth=None, seed=0):
    r"""Spatial bagging

    Bootstrap aggregation of CART regression trees (Breiman 1996) on the
    spatial features ``[X, s1, s2]``: the random forest of
    :func:`morie.fn.zxsrf.spatial_rf` with every feature tried at every split
    (``mtry = p``), bootstrap samples from Philox draws.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    n_trees : int
        Trees.
    min_leaf : int
        Minimum leaf size.
    max_depth : int, optional
        Depth cap.
    seed : int
        Philox seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the out-of-bag RMSE; ``extra`` has ``oob`` and ``fitted``.

    References
    ----------
    Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123-140.

    Hengl, T., Nussbaum, M., Wright, M. N., Heuvelink, G. B. M. and Graeler, B. (2018). Random forest
    as a generic framework for predictive modeling of spatial and spatio-temporal variables. *PeerJ*, 6, e5518.

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> round(spatial_bagging(y, X, S, n_trees=20, min_leaf=3, seed=1).value, 10)
    0.694697733
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, 1)
    trees, oob = sm.forest(Z, yv, int(n_trees), None, int(min_leaf), max_depth, seed)
    ok = [i for i, v in enumerate(oob) if v == v]
    return DescriptiveResult(
        name="zxsbg",
        value=sm.rmse([yv[i] for i in ok], [oob[i] for i in ok]),
        extra={"oob": oob, "fitted": sm.forest_predict(trees, Z)},
    )


spat = spatial_bagging
spatialbagging = spatial_bagging


def cheatsheet() -> str:
    return "spatial_bagging(...) -> Spatial bagging"
