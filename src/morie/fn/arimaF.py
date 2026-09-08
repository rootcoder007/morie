# morie.fn -- function file (rootcoder007/morie)
"""Conditional sum of squares for an ARIMA(p, d, q) model."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["arimacss", "arima"]


def arimacss(y, phi=(), theta=(), d=0, mu=0.0):
    """Residuals, CSS and Gaussian log-likelihood at given ARIMA parameters.

    The Box-Jenkins model writes a differenced series w_t = (1 - B)^d y_t
    as an ARMA(p, q),

        (1 - phi_1 B - ... - phi_p B^p)(w_t - mu)
            = (1 + theta_1 B + ... + theta_q B^q) e_t,

    so the innovations are recovered by the recursion

        e_t = (w_t - mu) - sum_i phi_i (w_{t-i} - mu)
                         - sum_j theta_j e_{t-j},

    started with zeros for the presample values -- the *conditional* sum
    of squares.  Summing e_t^2 over the m = n - d - p usable terms gives
    CSS, and the conditional Gaussian log-likelihood at the profiled
    innovation variance sigma^2 = CSS/m is

        logLik = -(m/2)(1 + log(2 pi) + log(CSS/m)).

    This is an evaluation, not a fit: it returns the objective an
    optimiser would minimise, which keeps the routine free of any
    iteration count or convergence tolerance.

    Parameters
    ----------
    y : array-like
        Observed series.
    phi : sequence
        Autoregressive coefficients, length p.
    theta : sequence
        Moving-average coefficients in the sign convention above (the
        Box-Jenkins +theta convention), length q.
    d : int
        Order of differencing.
    mu : float
        Mean of the differenced series.

    Returns
    -------
    RichResult
        ``css``, ``sigma2``, ``loglik``, ``aic``, ``resid``, ``diff``,
        ``m``, ``p``, ``q``, ``d``, ``n``.

    References
    ----------
    Box, G. E. P. and Jenkins, G. M. (1970), Time Series Analysis:
    Forecasting and Control, Holden-Day, Chapters 4 and 7: Chapter 4
    defines the ARIMA(p,d,q) operator equation above and Chapter 7 the
    conditional least-squares estimate obtained by starting the
    recursion from zero presample innovations.  Standard published form;
    the monograph was not in the local corpus and was not read for this
    implementation.
    """
    y = C.vec(y)
    n = len(y)
    ph = [float(v) for v in phi]
    th = [float(v) for v in theta]
    p, q = len(ph), len(th)
    d = int(d)
    if d < 0:
        raise ValueError("d must be non-negative")
    if n <= d + p:
        raise ValueError("series too short for the requested orders")
    w = list(y)
    for _ in range(d):
        w = [w[i + 1] - w[i] for i in range(len(w) - 1)]
    mu = float(mu)
    z = [v - mu for v in w]
    nw = len(z)
    e = [0.0] * nw
    css = 0.0
    m = 0
    for t in range(p, nw):
        v = z[t]
        for i in range(p):
            v -= ph[i] * z[t - 1 - i]
        for j in range(q):
            if t - 1 - j >= 0:
                v -= th[j] * e[t - 1 - j]
        e[t] = v
        css += v * v
        m += 1
    s2 = css / m
    ll = -0.5 * m * (1.0 + math.log(2.0 * math.pi) + math.log(s2))
    k = p + q + 1
    return RichResult(payload={
        "css": css, "sigma2": s2, "loglik": ll,
        "aic": -2.0 * ll + 2.0 * k, "resid": e[p:], "diff": w,
        "m": m, "p": p, "q": q, "d": d, "n": n,
        "method": "ARIMA conditional sum of squares (Box-Jenkins 1970)"})


arima = arimacss


def cheatsheet():
    return "arimaF: Conditional sum of squares for an ARIMA(p, d, q) model."
