# morie.fn -- function file (rootcoder007/morie)
"""Spatial Poisson fixed-effects panel."""

from __future__ import annotations

from ._containers import SpatialResult
from .scpboot import _fit


def scpflx(y, X, W, unit_id):
    r"""Spatial Poisson fixed-effects panel.

    The spatial-lag Poisson model ``E y_t = exp((I - rho W)^{-1} (X_t beta
    + alpha))`` for ``T`` periods of ``N`` units with unit fixed effects
    ``alpha`` (one dummy per unit, which absorbs the intercept), fitted by
    profile maximum likelihood (:func:`morie.fn.spcount.sar_poisson`, then
    Newton steps on the analytic profile score) on
    the stacked data with block-diagonal weights ``I_T (x) W`` (Lambert,
    Brown and Florax 2010; Elhorst 2014, ch. 3 for the panel stacking).
    Observations are stacked period by period (the first ``N`` rows are
    period 1); constant columns of ``X`` are dropped.

    Parameters
    ----------
    y : array-like, shape (N T,)
        Counts, stacked by period.
    X : array-like, shape (N T, p)
        Time-varying regressors.
    W : array-like, shape (N, N)
        Spatial weights among the units.
    unit_id : array-like, shape (N T,)
        Unit label of every observation.

    Returns
    -------
    SpatialResult
        ``statistic`` is ``rho``; ``extra`` has ``beta`` (non-constant
        regressors), ``unit_effects``, ``loglik`` and ``fitted``.

    References
    ----------
    Lambert, D. M., Brown, J. P. and Florax, R. J. G. M. (2010). A two-step estimator for a spatial lag
    model of counts. *Regional Science and Urban Economics*, 40(4), 241-252.

    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> X = [[0.1], [0.5], [0.3], [0.4], [0.9], [0.2]]
    >>> r = scpflx([1, 4, 2, 2, 6, 1], X, W, ["a", "b", "c", "a", "b", "c"])
    >>> round(r.statistic, 6), [round(b, 6) for b in r.extra["beta"]]
    (-0.197577, [1.345387])
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Xm = [
        [float(v) for v in (r if isinstance(r, (list, tuple)) else [r])]
        for r in (X.tolist() if hasattr(X, "tolist") else X)
    ]
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    ids = list(unit_id.tolist() if hasattr(unit_id, "tolist") else unit_id)
    n, N = len(yv), len(Wm)
    if n % N or len(Xm) != n or len(ids) != n:
        raise ValueError("y, X and unit_id must have N T rows, stacked by period")
    T = n // N
    keep = [k for k in range(len(Xm[0])) if max(r[k] for r in Xm) > min(r[k] for r in Xm)]
    units = []
    for u in ids:
        if u not in units:
            units.append(u)
    Z = [[Xm[i][k] for k in keep] + [1.0 if ids[i] == u else 0.0 for u in units] for i in range(n)]
    Wf = [[Wm[i % N][j % N] if i // N == j // N else 0.0 for j in range(n)] for i in range(n)]
    rho, g = _fit(yv, Z, Wf)
    b = list(g["coefficients"])
    return SpatialResult(
        name="scpflx",
        statistic=rho,
        extra={
            "beta": b[: len(keep)],
            "unit_effects": dict(zip(units, b[len(keep) :])),
            "loglik": g["loglik"],
            "fitted": list(g["fitted"]),
            "periods": T,
        },
    )


scpflx_fn = scpflx


def cheatsheet() -> str:
    return "scpflx(y, X, W, unit_id) -> SAR-Poisson with unit fixed effects (stacked panel)"
