"""Spatial Durbin model (SDM)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ._schab_rho import safe_search_interval

from ._containers import SpatialResult


def spatial_durbin_model(
    Z: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
) -> SpatialResult:
    r"""Fit a Spatial Durbin Model.

    .. math::

        Z = \rho W Z + X \beta + W X \theta + \varepsilon

    Parameters
    ----------
    Z : np.ndarray
        Response, shape ``(n,)``.
    X : np.ndarray
        Covariates, shape ``(n, p)``.
    W : np.ndarray
        Spatial weights, shape ``(n, n)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is estimated :math:`\rho`.
        ``extra`` has ``beta``, ``theta``, ``residuals``.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009) Introduction to Spatial
    Econometrics. Chapman and Hall/CRC. doi:10.1201/9781420064254
    Bivand, R. S., Pebesma, E., and Gomez-Rubio, V. (2013) Applied
    Spatial Data Analysis with R, 2nd ed., Springer. Sec. 9.4.2
    "Spatial Econometrics Approaches", pp. 307-311.
    NOT in Schabenberger & Gotway (2005): "Durbin" appears there only
    in the reference list.

    .. epigraph::

    """
    Z = np.asarray(Z, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = len(Z)
    I = np.eye(n)

    WX = W @ X
    X_aug = np.column_stack([X, WX])

    lo, hi = safe_search_interval(W, "identity")

    def neg_ll(rho):
        A = I - rho * W
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return np.inf
        Zy = A @ Z
        coef = np.linalg.lstsq(X_aug, Zy, rcond=None)[0]
        resid = Zy - X_aug @ coef
        sigma2 = np.sum(resid**2) / n
        if sigma2 <= 0:
            return np.inf
        return -(-0.5 * n * np.log(2 * np.pi * sigma2) + logdet - 0.5 * n)

    opt = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10 * max(hi - lo, 1.0)})
    best_rho = float(opt.x) if np.isfinite(neg_ll(opt.x)) else 0.0
    A = I - best_rho * W
    Zy = A @ Z
    best_coef = np.linalg.lstsq(X_aug, Zy, rcond=None)[0]
    best_resid = Zy - X_aug @ best_coef
    _s, _ld = np.linalg.slogdet(A)
    best_ll = (-0.5 * n * np.log(2 * np.pi * np.sum(best_resid**2) / n)
               + _ld - 0.5 * n)

    p = X.shape[1]
    if best_coef is None:
        best_coef = np.zeros(2 * p)
        best_resid = Z.copy()

    return SpatialResult(
        name="spatial_durbin_model",
        statistic=float(best_rho),
        p_value=None,
        extra={
            "beta": best_coef[:p],
            "theta": best_coef[p:],
            "residuals": best_resid,
            "log_likelihood": float(best_ll),
        },
    )


sgdbn = spatial_durbin_model


def cheatsheet() -> str:
    return "spatial_durbin_model({}) -> Spatial Durbin model (SDM)."
