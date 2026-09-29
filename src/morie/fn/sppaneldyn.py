# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel data models (Elhorst 2014): fixed-effects spatial lag, error and Durbin models
by concentrated maximum likelihood, the random-effects spatial lag model, and the dynamic
spatial panel (spatial ARX) with time- and space-time-lagged responses."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import solve, ssum
from ._richresult import RichResult
from .sppanel import spatial_panel_ml

__all__ = ["sp_panel_dynamic"]


def _matvec(M, v):
    return [ssum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _eigs(W):
    return [complex(v) for v in np.linalg.eigvals(np.asarray(W, dtype=float))]


def _ld_eig(ev, p):
    # ln|I - p W| = sum ln|1 - p lambda_i| (Ord 1975)
    return ssum(math.log(abs(1.0 - p * v)) for v in ev)


def _tr_eig(ev, p):
    # tr(W (I - p W)^(-1)) = sum lambda_i / (1 - p lambda_i)
    return ssum((v / (1.0 - p * v)).real for v in ev)


def _refine(dfun, x0, lo, hi, width=1e-6):
    # bisection on the score in a small bracket around the golden-section optimum
    a, b = max(lo, x0 - width), min(hi, x0 + width)
    fa, fb = dfun(a), dfun(b)
    if fa * fb > 0:
        return x0
    for _ in range(100):
        m = (a + b) / 2.0
        if m in (a, b):
            break
        fm = dfun(m)
        if fm == 0.0:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2.0


def _ols(X, y):
    k = len(X[0])
    XtX = [[ssum(r[a] * r[b] for r in X) for b in range(k)] for a in range(k)]
    Xty = [ssum(r[a] * v for r, v in zip(X, y)) for a in range(k)]
    b = solve(XtX, Xty)
    e = [v - ssum(r[j] * b[j] for j in range(k)) for r, v in zip(X, y)]
    return b, e, XtX


def _demean(v, T, N, effects):
    # v stacked time-major: v[t * N + i]
    out = list(v)
    if effects in ("individual", "twoways"):
        um = [ssum(v[t * N + i] for t in range(T)) / T for i in range(N)]
        out = [out[t * N + i] - um[i] for t in range(T) for i in range(N)]
    if effects in ("time", "twoways"):
        tm = [ssum(v[t * N + i] for i in range(N)) / N for t in range(T)]
        out = [out[t * N + i] - tm[t] for t in range(T) for i in range(N)]
        if effects == "twoways":
            g = ssum(v) / (T * N)
            out = [a + g for a in out]
    return out


def _golden(f, lo, hi, tol=1e-12, max_iter=300):
    r = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _stack(y, X, W, lagx):
    T, N = len(y), len(y[0])
    K = len(X[0][0]) if X is not None and len(X[0][0]) else 0
    ys = [float(y[t][i]) for t in range(T) for i in range(N)]
    wy = [v for t in range(T) for v in _matvec(W, [float(a) for a in y[t]])]
    cols = [[float(X[t][i][k]) for t in range(T) for i in range(N)] for k in range(K)]
    if lagx:
        for k in range(K):
            cols.append([v for t in range(T) for v in _matvec(W, [float(X[t][i][k]) for i in range(N)])])
    return T, N, ys, wy, cols


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
