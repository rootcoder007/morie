# morie.fn -- function file (rootcoder007/morie)
"""Hannan-Rissanen estimation of an ARIMA(p, d, q) model."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["arimahr", "arima_box_jenkins"]


def _difference(y, d):
    w = list(y)
    for _ in range(d):
        w = [w[i + 1] - w[i] for i in range(len(w) - 1)]
    return w


def arimahr(y, p=1, q=1, d=0, m=None):
    """Two-stage least-squares estimates of the ARMA coefficients.

    Estimating a mixed model directly is non-linear because the moving
    average involves unobserved innovations.  Hannan and Rissanen remove
    the non-linearity by producing proxies for them first:

      1. fit a long autoregression of order m by ordinary least squares
         and keep its residuals ehat_t as estimates of the innovations;
      2. regress w_t on its own p lags and on the m lagged ehat, giving
         phi and theta in one linear step.

    The whole procedure is two ordinary least-squares solves, so it is
    closed form -- no iteration count, no convergence tolerance, and the
    same answer in every arm.

    Parameters
    ----------
    y : array-like
        Observed series.
    p, q : int
        Autoregressive and moving-average orders.
    d : int
        Order of differencing applied before fitting.
    m : int or None
        Order of the long autoregression in stage one.  ``None`` uses
        max(p + q + 1, ceil(sqrt(len(w)))), a common default.

    Returns
    -------
    RichResult
        ``phi``, ``theta``, ``intercept``, ``sigma2``, ``resid``,
        ``m``, ``p``, ``q``, ``d``, ``nobs``.

    References
    ----------
    Hannan, E. J. and Rissanen, J. (1982), "Recursive estimation of mixed
    autoregressive-moving average order", Biometrika 69(1), 81-94, which
    is the two-stage regression above; the ARIMA(p,d,q) model itself is
    Box, Jenkins and Reinsel (1994), Time Series Analysis: Forecasting
    and Control, 3rd edn, Prentice Hall, Chapter 4.  Standard published
    form; neither source was in the local corpus and neither was read for
    this implementation.
    """
    y = C.vec(y)
    p, q, d = int(p), int(q), int(d)
    if p < 0 or q < 0 or d < 0:
        raise ValueError("orders must be non-negative")
    w = _difference(y, d)
    nw = len(w)
    if m is None:
        m = max(p + q + 1, int(nw ** 0.5) + 1)
    m = int(m)
    if nw <= m + max(p, q) + 1:
        raise ValueError("series too short for the requested orders")
    Xa = [[1.0] + [w[t - 1 - i] for i in range(m)] for t in range(m, nw)]
    ya = [w[t] for t in range(m, nw)]
    fa = C.lstsq(Xa, ya)
    eh = [0.0] * nw
    for k, t in enumerate(range(m, nw)):
        eh[t] = fa["resid"][k]
    s = m + max(p, q)
    Xb = []
    yb = []
    for t in range(s, nw):
        row = [1.0]
        row += [w[t - 1 - i] for i in range(p)]
        row += [eh[t - 1 - j] for j in range(q)]
        Xb.append(row)
        yb.append(w[t])
    fb = C.lstsq(Xb, yb)
    b = fb["beta"]
    res = fb["resid"]
    nobs = len(yb)
    k = p + q + 1
    return RichResult(payload={
        "phi": b[1:1 + p], "theta": b[1 + p:1 + p + q], "intercept": b[0],
        "sigma2": sum(v * v for v in res) / max(nobs - k, 1),
        "resid": res, "m": m, "p": p, "q": q, "d": d, "nobs": nobs,
        "method": "Hannan-Rissanen ARMA estimation (Hannan-Rissanen 1982)"})


arima_box_jenkins = arimahr


def cheatsheet():
    return "arimab: Hannan-Rissanen estimation of an ARIMA(p, d, q) model."
