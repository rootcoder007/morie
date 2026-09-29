# morie.fn -- function file (rootcoder007/morie)
"""Quantile regression by linear programming and the two-stage spatial autoregressive quantile
regression of Kim and Muller (2004)."""

from __future__ import annotations

from . import _array_core as np
from ._qpcore import simplex_standard, ssum
from ._richresult import RichResult

__all__ = ["quantile_regression_lp", "spatial_quantile_iv"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def quantile_regression_lp(y, X, tau=0.5):
    r"""Regression quantile of Koenker and Bassett (1978) as a linear programme.

    ``min tau 1'u+ + (1 - tau) 1'u-`` subject to ``X b+ - X b- + u+ - u- = y``
    and all variables non-negative, solved by the two-phase simplex with
    Bland's rule; ``b = b+ - b-``. ``X`` should include the intercept column.
    For continuous data the solution is unique (then equal to
    ``quantreg::rq``).

    References
    ----------
    Koenker, R. and Bassett, G. (1978). Regression quantiles. *Econometrica*
    46, 33-50.

    Examples
    --------
    >>> r = quantile_regression_lp([1.0, 3.0, 2.0, 5.0, 4.0], [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]])
    >>> [round(b, 12) for b in r.coefficients]
    [1.0, 0.75]
    """
    yv, Xm = _vec(y), _mat(X)
    n, k = len(Xm), len(Xm[0])
    A = [
        Xm[i]
        + [-v for v in Xm[i]]
        + [1.0 if j == i else 0.0 for j in range(n)]
        + [-1.0 if j == i else 0.0 for j in range(n)]
        for i in range(n)
    ]
    c = [0.0] * (2 * k) + [float(tau)] * n + [1.0 - float(tau)] * n
    x, status = simplex_standard(c, A, yv, tol=1e-10, max_iter=100000)
    if status != "optimal":
        raise RuntimeError("linear programme " + status)
    b = [x[a] - x[k + a] for a in range(k)]
    res = [v - ssum(r[a] * b[a] for a in range(k)) for r, v in zip(Xm, yv)]
    obj = ssum(tau * e if e >= 0 else (tau - 1) * e for e in res)
    return RichResult(payload={"coefficients": b, "residuals": res, "objective": obj})


def spatial_quantile_iv(y, X, W, tau=0.5, Z=None):
    r"""Two-stage quantile regression for the spatial autoregressive model (Kim and Muller 2004).

    Stage 1: the regression quantile of ``W y`` on the instruments (default
    ``[1, X, W X]``); stage 2: the regression quantile of ``y`` on ``[1, X, fitted
    W y]``. The last coefficient is the spatial lag ``rho(tau)``; ``X`` without
    the intercept. McMillen's ``qregspiv`` point estimates.

    References
    ----------
    Kim, T.-H. and Muller, C. (2004). Two-stage quantile regression when the
    first stage is based on quantile regression. *Econometrics Journal* 7,
    218-231.

    Zietz, J., Zietz, E. N. and Sirmans, G. S. (2008). Determinants of house
    prices: a quantile regression approach. *Journal of Real Estate Finance and
    Economics* 37, 317-333.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
    >>> r = spatial_quantile_iv([1.0, 2.5, 2.0, 4.1, 3.2], [[0.1], [0.9], [0.4], [1.5], [1.1]], W)
    >>> len(r.coefficients), -1 < r.rho < 1
    (2, True)
    """
    yv, Xm, Wm = _vec(y), _mat(X), _mat(W)
    n, k = len(Xm), len(Xm[0])
    wy = [ssum(Wm[i][j] * yv[j] for j in range(n)) for i in range(n)]
    if Z is None:
        wx = [[ssum(Wm[i][j] * Xm[j][a] for j in range(n)) for a in range(k)] for i in range(n)]
        Zm = [[1.0] + Xm[i] + wx[i] for i in range(n)]
    else:
        Zm = [[1.0] + r for r in _mat(Z)]
    s1 = quantile_regression_lp(wy, Zm, tau)
    fitted = [v - e for v, e in zip(wy, s1.residuals)]
    s2 = quantile_regression_lp(yv, [[1.0] + Xm[i] + [fitted[i]] for i in range(n)], tau)
    b = s2.coefficients
    return RichResult(
        payload={"coefficients": b[: k + 1], "rho": b[k + 1], "first_stage": s1.coefficients, "wy_hat": fitted}
    )


def cheatsheet() -> str:
    return "quantile_regression_lp / spatial_quantile_iv -> quantile regression."
