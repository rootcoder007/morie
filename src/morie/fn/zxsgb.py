# morie.fn -- function file (rootcoder007/morie)
"""Spatial gradient boosting"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_gbm(y, X, coords, *, n_trees=100, rate=0.1, max_depth=2, min_leaf=5):
    r"""Spatial gradient boosting

    Least-squares gradient boosting (Friedman 2001) of regression trees on the
    spatial features ``[X, s1, s2]``: start at the mean, then repeatedly fit a
    depth-``max_depth`` CART tree to the residuals and add it shrunk by
    ``rate``. Deterministic (no subsampling).

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    n_trees : int
        Boosting rounds.
    rate : float
        Shrinkage.
    max_depth : int
        Tree depth.
    min_leaf : int
        Minimum leaf size.

    Returns
    -------
    DescriptiveResult
        ``value`` is the training RMSE; ``extra`` has ``fitted``, ``init`` and ``trees``.

    References
    ----------
    Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of
    Statistics*, 29(5), 1189-1232.

    Hengl, T., Nussbaum, M., Wright, M. N., Heuvelink, G. B. M. and Graeler, B. (2018). Random forest
    as a generic framework for predictive modeling of spatial and spatio-temporal variables. *PeerJ*, 6, e5518.

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> round(spatial_gbm(y, X, S, n_trees=30, min_leaf=3).value, 10)
    0.1632072569
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, 1)
    f0, trees, F = sm.boost(Z, yv, int(n_trees), float(rate), max_depth, int(min_leaf))
    return DescriptiveResult(
        name="zxsgb", value=sm.rmse(yv, F), extra={"fitted": F, "init": f0, "trees": trees, "rate": float(rate)}
    )


spat = spatial_gbm
spatialgbm = spatial_gbm


def cheatsheet() -> str:
    return "spatial_gbm(...) -> Spatial gradient boosting"
