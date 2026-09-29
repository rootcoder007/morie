# morie.fn -- function file (rootcoder007/morie)
"""Spatial LOO cross-validation"""

from __future__ import annotations

from . import _spml as sm
from ._containers import DescriptiveResult
from ._qpcore import inverse


def spatial_cv_loo(y, X, coords, *, trend=1):
    r"""Spatial LOO cross-validation

    Leave-one-out prediction error of the least-squares model on the spatial
    features ``[1, X, s1, s2]`` (a linear trend surface; ``trend=2`` adds the
    quadratic terms), computed exactly from the hat matrix: the deleted
    residual is ``e_i / (1 - h_ii)`` (Cook and Weisberg 1982). Plain LOO is
    optimistic under spatial autocorrelation because neighbours of the held-out
    point stay in the training set; see :func:`morie.fn.zxsbu.spatial_cv_buffer`
    and :func:`morie.fn.zxsbv.spatial_cv_block` (Roberts et al. 2017).

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p) or None
        Covariates (without intercept).
    coords : array-like, shape (n, 2)
        Locations; they enter as covariates (the spatial features).
    trend : int
        1 (linear) or 2 (quadratic) trend surface.

    Returns
    -------
    DescriptiveResult
        ``value`` is the LOO RMSE; ``extra`` has ``predictions`` (held-out) and ``leverage``.

    References
    ----------
    Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial,
    hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.

    Cressie, N. (1993). *Statistics for Spatial Data*, revised edition. Wiley, sec. 4.1 (trend surfaces).

    Examples
    --------
    >>> S = [[(i % 5) / 4, (i // 5) / 3] for i in range(20)]
    >>> X = [[((i * 7) % 11) / 10] for i in range(20)]
    >>> y = [1 + 2 * x[0] + s[0] - s[1] + ((i * 3) % 5 - 2) / 10 for i, (x, s) in enumerate(zip(X, S))]
    >>> round(spatial_cv_loo(y, X, S).value, 12)
    0.154172984333
    """
    yv = sm.vec(y)
    Z = sm.features(X, coords, trend)
    D = [[1.0] + r for r in Z]
    p = len(D[0])
    A = [[sm.math.fsum(r[a] * r[c] for r in D) for c in range(p)] for a in range(p)]
    Ai = inverse(A)
    b = sm.ols_fit(Z, yv)
    fit = sm.ols_predict(b, Z)
    h = [sm.math.fsum(r[a] * Ai[a][c] * r[c] for a in range(p) for c in range(p)) for r in D]
    pred = [t - (t - f) / (1.0 - hh) for t, f, hh in zip(yv, fit, h)]
    return DescriptiveResult(
        name="zxscv", value=sm.rmse(yv, pred), extra={"predictions": pred, "leverage": h, "n": len(yv)}
    )


spat = spatial_cv_loo
spatialcvloo = spatial_cv_loo


def cheatsheet() -> str:
    return "spatial_cv_loo(...) -> Spatial LOO cross-validation"
