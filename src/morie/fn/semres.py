# morie.fn -- function file (rootcoder007/morie)
"""SEM residual autocorrelation (filtered residuals)."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semres(resid, W, lam=0.3, X=None):
    r"""SEM residual autocorrelation (filtered residuals).

    Moran's I of the residuals of a spatial error model, ``I = (n / S0) e'We / e'e``
    (Cliff and Ord 1981). With the design ``X`` the exact moments under
    normality for OLS-type residuals ``e = M y``, ``M = I - X(X'X)^{-1}X'``,
    are added -- ``E(I) = (n/S0) tr(MW)/(n-k)`` and the Cliff-Ord variance --
    with the one-sided (greater) normal p-value, as ``spdep::lm.morantest``
    (:func:`morie.fn.spdurbin.residual_moran`).
    The residuals are first filtered, ``e* = (I - lambda W) e`` (and
    ``X* = (I - lambda W) X`` when ``X`` is given), so the check is on the
    innovations that the fitted error process should have whitened.

    Parameters
    ----------
    resid : array-like, shape (n,)
        Model residuals.
    W : array-like, shape (n, n)
        Spatial weights.
    lam : float
        Error autoregressive parameter of the filter.
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
    >>> round(semres(e, W, 0.2).statistic, 12)
    -0.8430335097
    """
    Wm = sd._mat(W)
    e = sd._vec(resid)
    e = [a - float(lam) * b for a, b in zip(e, sd._mv(Wm, e))]
    if X is not None:
        Xm = sd._mat(X)
        X = [[a - float(lam) * b for a, b in zip(r, s)] for r, s in zip(Xm, sd._mm(Wm, Xm))]
    r = sd.resid_moran(e, Wm, X)
    return SpatialResult(
        name="semres",
        statistic=r["I"],
        p_value=r["p_value"],
        expected=r["expected"],
        variance=r["variance"],
        extra={"z": r["z"]},
    )


semres_fn = semres


def cheatsheet() -> str:
    return "semres(resid, W, lam=0.3, X=None) -> Moran's I of residuals (exact test with X)"
