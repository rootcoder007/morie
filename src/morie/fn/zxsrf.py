# morie.fn -- function file (rootcoder007/morie)
"""Spatial random forest"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_rf(y, X, coords, *, n_trees=100, mtry=None, min_leaf=5, max_depth=None, seed=0):
    r"""Spatial random forest

    Random forest regression (Breiman 2001) with the coordinates as
    covariates -- the simplest form of spatial random forest (Hengl et al.
    2018): ``n_trees`` CART trees, each grown on a bootstrap sample (Philox
    draws, ``seed``) with ``mtry`` features tried per split (default
    ``max(1, p // 3)``), leaves of at least ``min_leaf``; predictions are tree
    averages and the out-of-bag predictions give the error estimate.

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
    mtry : int, optional
        Features per split.
    min_leaf : int
        Minimum leaf size.
    max_depth : int, optional
        Depth cap.
    seed : int
        Philox seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the out-of-bag RMSE; ``extra`` has ``oob``, ``fitted`` and ``trees``.

    References
    ----------
    Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

    Hengl, T., Nussbaum, M., Wright, M. N., Heuvelink, G. B. M. and Graeler, B. (2018). Random forest
    as a generic framework for predictive modeling of spatial and spatio-temporal variables. *PeerJ*, 6, e5518.

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> r = spatial_rf(y, X, S, n_trees=20, min_leaf=3, seed=1)
    >>> round(r.value, 10)
    0.5895014499
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, 1)
    m = max(1, len(Z[0]) // 3) if mtry is None else int(mtry)
    trees, oob = sm.forest(Z, yv, int(n_trees), m, int(min_leaf), max_depth, seed)
    ok = [i for i, v in enumerate(oob) if v == v]
    return DescriptiveResult(
        name="zxsrf",
        value=sm.rmse([yv[i] for i in ok], [oob[i] for i in ok]),
        extra={"oob": oob, "fitted": sm.forest_predict(trees, Z), "trees": trees, "mtry": m},
    )


spat = spatial_rf
spatialrf = spatial_rf


def cheatsheet() -> str:
    return "spatial_rf(...) -> Spatial random forest"
