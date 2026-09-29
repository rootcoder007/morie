# morie.fn -- function file (rootcoder007/morie)
"""SDM residual autocorrelation check."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmres(resid, W, X=None):
    r"""SDM residual autocorrelation check.

    Moran's I of the residuals of a spatial Durbin model, ``I = (n / S0) e'We / e'e``
    (Cliff and Ord 1981). With the design ``X`` the exact moments under
    normality for OLS-type residuals ``e = M y``, ``M = I - X(X'X)^{-1}X'``,
    are added -- ``E(I) = (n/S0) tr(MW)/(n-k)`` and the Cliff-Ord variance --
    with the one-sided (greater) normal p-value, as ``spdep::lm.morantest``
    (:func:`morie.fn.spdurbin.residual_moran`).
    Parameters
    ----------
    resid : array-like, shape (n,)
        Model residuals.
    W : array-like, shape (n, n)
        Spatial weights.
    X : array-like, shape (n, k), optional
        Design whose column space the residuals are orthogonal to.

    Returns
    -------
    SpatialResult
        ``statistic`` is I; with ``X`` also ``expected``, ``variance``,
        ``p_value`` and ``extra["z"]``.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion, London.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> e = [0.3, -0.2, 0.5, -0.4, 0.1, -0.3]
    >>> round(sdmres(e, W).statistic, 12)
    -0.7421875
    """
    r = sd.resid_moran(resid, W, X)
    return SpatialResult(
        name="sdmres",
        statistic=r["I"],
        p_value=r["p_value"],
        expected=r["expected"],
        variance=r["variance"],
        extra={"z": r["z"]},
    )


sdmres_fn = sdmres


def cheatsheet() -> str:
    return "sdmres(resid, W, X=None) -> Moran's I of residuals (exact test with X)"
