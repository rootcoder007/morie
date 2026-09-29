# morie.fn -- function file (rootcoder007/morie)
"""Spatial dynamic panel (lagged dependent variable)."""

from .sppaneldyn import sp_panel_dynamic
from .sppfe import _stack


def sppdyn(y, X, W, time_id, unit_id, space_time_lag=True, effects="individual"):
    r"""Dynamic spatial panel y_t = rho W y_t + gamma y_{t-1} + delta W y_{t-1} + X_t beta + mu + e_t.

    Conditional on the first period, y_{t-1} (and W y_{t-1} when
    space_time_lag) enter the fixed-effects spatial lag model as
    regressors (Elhorst 2014, ch. 4; the within estimator carries the Nickell
    bias of order 1/T, Yu, de Jong and Lee 2008). Long-format rows are
    reshaped by (period, unit id); constant columns of X are dropped. Thin
    front-end to :func:`morie.fn.sppaneldyn.sp_panel_dynamic`; beta lists
    gamma, delta (if used) and the X coefficients.

    References
    ----------
    Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data
    to Spatial Panels*. Springer.
    Yu, J., de Jong, R. and Lee, L.-F. (2008). Quasi-maximum likelihood
    estimators for spatial dynamic panel data with fixed effects when both n
    and T are large. *Journal of Econometrics* 146, 118-134.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> [round(b, 8) for b in sppdyn(y, X, W, tid, uid)["beta"]]
    [-0.08559497, 0.11230784, 0.85364784]
    """
    yv, Xm, N, T = _stack(y, X, time_id, unit_id)
    Y = [yv[t * N : (t + 1) * N] for t in range(T)]
    XX = [[Xm[t * N + i] for i in range(N)] for t in range(T)]
    return sp_panel_dynamic(Y, XX, W, effects=effects, space_time_lag=space_time_lag)


sppdyn_fn = sppdyn


def cheatsheet() -> str:
    return "sppdyn(y, X, W, time_id, unit_id) -> dynamic spatial panel (y_{t-1}, W y_{t-1} as regressors)."
