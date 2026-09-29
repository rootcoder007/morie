# morie.fn -- function file (rootcoder007/morie)
"""Buffered spatial CV"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult


def spatial_cv_buffer(y, X, coords, radius, *, trend=1):
    r"""Buffered spatial CV

    Spatial (buffered) leave-one-out cross-validation (Le Rest et al. 2014):
    each observation is predicted by the least-squares model on the spatial
    features ``[1, X, s1, s2]`` refitted without every observation within
    distance ``radius`` of it, so spatially autocorrelated neighbours cannot
    leak into the training set. ``radius = 0`` is ordinary LOO.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    radius : float
        Buffer radius.
    trend : int
        1 (linear) or 2 (quadratic) trend surface.

    Returns
    -------
    DescriptiveResult
        ``value`` is the buffered-LOO RMSE; ``extra`` has ``predictions`` and ``n_train`` per point.

    References
    ----------
    Le Rest, K., Pinaud, D., Monestiez, P., Chadoeuf, J. and Bretagnolle, V. (2014). Spatial
    leave-one-out cross-validation for variable selection in the presence of spatial autocorrelation.
    *Global Ecology and Biogeography*, 23(7), 811-820.

    Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial,
    hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> round(spatial_cv_buffer(y, X, S, 0.3).value, 12)
    0.123917430083
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, trend)
    C = sm.mat(coords)
    n = len(yv)
    pred, ntr = [], []
    for i in range(n):
        tr = [j for j in range(n) if sm.math.dist(C[i][:2], C[j][:2]) > radius]
        if len(tr) <= len(Z[0]) + 1:
            raise ValueError("buffer leaves too few training points")
        b = sm.ols_fit([Z[j] for j in tr], [yv[j] for j in tr])
        pred.append(sm.ols_predict(b, [Z[i]])[0])
        ntr.append(len(tr))
    return DescriptiveResult(
        name="zxsbu", value=sm.rmse(yv, pred), extra={"predictions": pred, "n_train": ntr, "radius": float(radius)}
    )


spat = spatial_cv_buffer


def cheatsheet() -> str:
    return "spatial_cv_buffer(...) -> Buffered spatial CV"
