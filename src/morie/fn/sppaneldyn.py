# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel data models (Elhorst 2014): fixed-effects spatial lag, error and Durbin models
by concentrated maximum likelihood, the random-effects spatial lag model, and the dynamic
spatial panel (spatial ARX) with time- and space-time-lagged responses."""

from __future__ import annotations

from ._qpcore import ssum
from ._richresult import RichResult
from .sppanel import spatial_panel_ml

__all__ = ["sp_panel_dynamic"]


def _matvec(M, v):
    return [ssum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def sp_panel_dynamic(
    y, X, W, *, effects: str = "individual", space_time_lag: bool = True, bounds=(-0.99, 0.99)
) -> RichResult:
    r"""Dynamic spatial panel (spatial ARX): ``y_t = rho W y_t + gamma y_{t-1} + delta W y_{t-1} + X_t beta + mu + e_t``.

    Conditional on the first period, the lagged response ``y_{t-1}`` (and,
    when ``space_time_lag``, its spatial lag ``W y_{t-1}``) join ``X_t`` as
    regressors of the fixed-effects spatial lag model
    :func:`morie.fn.sppanel.spatial_panel_ml` (Elhorst 2014, ch. 4; the
    within estimator carries the Nickell bias of order ``1/T``, which Yu, de
    Jong and Lee 2008 correct for). ``y`` is ``T x N`` (rows are periods) and
    ``X`` is ``T x N x K``. ``beta`` lists ``gamma``, ``delta`` (if used), then
    the ``X`` coefficients.

    References
    ----------
    Elhorst, J. P. (2014). *Spatial Econometrics*, ch. 4. Springer.
    Yu, J., de Jong, R. and Lee, L.-F. (2008). Quasi-maximum likelihood
    estimators for spatial dynamic panel data with fixed effects when both n
    and T are large. *Journal of Econometrics*, 146(1), 118-134.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [[1.0, 2.0, 1.5], [2.0, 2.5, 1.0], [1.5, 3.0, 2.5], [2.5, 2.0, 3.0], [2.0, 1.0, 2.2]]
    >>> X = [[[0.5], [1.0], [0.2]], [[1.5], [0.7], [0.1]], [[0.4], [2.0], [1.1]], [[1.2], [0.3], [1.9]], [[0.3], [0.9], [1.4]]]
    >>> len(sp_panel_dynamic(y, X, W).beta)
    3
    """
    T, N = len(y), len(y[0])
    Wf = [[float(v) for v in row] for row in W]
    yv, Xm = [], []
    for t in range(1, T):
        prev = [float(a) for a in y[t - 1]]
        wprev = _matvec(Wf, prev)
        for i in range(N):
            yv.append(float(y[t][i]))
            Xm.append([prev[i]] + ([wprev[i]] if space_time_lag else []) + [float(v) for v in X[t][i]])
    r = spatial_panel_ml(yv, Xm, Wf, N, model="lag", effects=effects, interval=bounds)
    return RichResult(
        payload={
            "rho": r["rho"],
            "beta": r["coefficients"],
            "coefficients": r["coefficients"],
            "sigma2": r["sigma2"],
            "loglik": r["loglik"],
            "residuals": r["residuals"],
            "n_obs": r["n_obs"],
        }
    )


def cheatsheet() -> str:
    return "sp_panel_dynamic -> dynamic spatial panel (spatial ARX) on the fixed-effects spatial lag model."
