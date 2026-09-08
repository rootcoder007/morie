# morie.fn -- function file (rootcoder007/morie)
"""ARIMAX: exogenous regression with ARMA errors."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["arimaxhr", "arimax"]


def arimaxhr(y, X, p=1, q=1, d=0, m=None):
    """Regression on exogenous variables with Hannan-Rissanen ARMA errors.

    An ARIMAX model adds a linear term in observed covariates to the
    Box-Jenkins structure,

        (1 - B)^d y_t = X_t beta + n_t,
        phi(B) n_t    = theta(B) e_t,

    so the covariate effect is estimated on the differenced scale and the
    dynamics are carried entirely by the disturbance n_t.  Fitting is two
    stages of ordinary least squares: regress the differenced response on
    the differenced covariates, then apply the Hannan-Rissanen two-stage
    regression to the residual series n_t.  Everything is closed form.

    Parameters
    ----------
    y : array-like
        Observed response.
    X : array-like, shape (n, r)
        Exogenous regressors, differenced internally to match y.
    p, q : int
        ARMA orders for the disturbance.
    d : int
        Order of differencing.
    m : int or None
        Stage-one long-autoregression order; ``None`` as in the ARMA case.

    Returns
    -------
    RichResult
        ``beta``, ``phi``, ``theta``, ``intercept``, ``sigma2``,
        ``noise``, ``resid``, ``p``, ``q``, ``d``, ``r``, ``nobs``.

    References
    ----------
    Box, G. E. P. and Jenkins, G. M. (1976), Time Series Analysis:
    Forecasting and Control, revised edn, Holden-Day, Chapters 10-11 on
    transfer-function models, of which the linear-regression-with-ARMA-
    errors form used here is the static special case; the disturbance is
    estimated by Hannan and Rissanen (1982), Biometrika 69(1), 81-94.
    Standard published form; neither source was in the local corpus and
    neither was read for this implementation.
    """
    y = C.vec(y)
    Xm = C.mat(X)
    n = len(y)
    if len(Xm) != n:
        raise ValueError("X must have one row per observation")
    r = len(Xm[0])
    p, q, d = int(p), int(q), int(d)
    if p < 0 or q < 0 or d < 0:
        raise ValueError("orders must be non-negative")
    w = list(y)
    Xd = [row[:] for row in Xm]
    for _ in range(d):
        w = [w[i + 1] - w[i] for i in range(len(w) - 1)]
        Xd = [[Xd[i + 1][j] - Xd[i][j] for j in range(r)]
              for i in range(len(Xd) - 1)]
    Xr = [[1.0] + row for row in Xd]
    f1b, f1f, nz, f1x = C.lstsq(Xr, w)
    nw = len(nz)
    if m is None:
        m = max(p + q + 1, int(nw ** 0.5) + 1)
    m = int(m)
    if nw <= m + max(p, q) + 1:
        raise ValueError("series too short for the requested orders")
    Xa = [[1.0] + [nz[t - 1 - i] for i in range(m)] for t in range(m, nw)]
    fa_b, fa_f, fa_r, fa_x = C.lstsq(Xa, [nz[t] for t in range(m, nw)])
    eh = [0.0] * nw
    for k, t in enumerate(range(m, nw)):
        eh[t] = fa_r[k]
    s = m + max(p, q)
    Xb = []
    yb = []
    for t in range(s, nw):
        row = [1.0]
        row += [nz[t - 1 - i] for i in range(p)]
        row += [eh[t - 1 - j] for j in range(q)]
        Xb.append(row)
        yb.append(nz[t])
    b, fb_f, res, fb_x = C.lstsq(Xb, yb)
    nobs = len(yb)
    k = p + q + 1 + r
    return RichResult(payload={
        "beta": f1b[1:], "phi": b[1:1 + p],
        "theta": b[1 + p:1 + p + q], "intercept": f1b[0],
        "sigma2": sum(v * v for v in res) / max(nobs - k, 1),
        "noise": nz, "resid": res, "p": p, "q": q, "d": d, "r": r,
        "nobs": nobs,
        "method": "ARIMAX by OLS plus Hannan-Rissanen errors (Box-Jenkins 1976)"})


arimax = arimaxhr


def cheatsheet():
    return "arimax: ARIMAX: exogenous regression with ARMA errors."
