# morie.fn -- function file (rootcoder007/morie)
"""Ratio estimator for a mean or total."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ratioest", "survey_ratio"]


def ratioest(y, x, X=None, N=float("inf"), level=0.95):
    """Ratio estimate of a mean or total, using an auxiliary variable.

    The ratio estimator beats the simple expansion estimator exactly
    when y is roughly proportional to x through the origin -- and is
    worse when it is not, which is why the residual variance
    sum (y_i - Rhat x_i)^2 is what appears in its variance rather than
    the variance of y.  Note the estimator is biased; the bias is
    O(1/n) and is not corrected here, matching Cochran's Chapter 6
    treatment.

    Formula: Rhat = ybar/xbar;  Yhat_R = Rhat X;
             v(Rhat) = (1 - f)/(n xbar^2) * sum (y_i - Rhat x_i)^2/(n - 1)

    Parameters
    ----------
    y : array-like
        Sample values of the variable of interest.
    x : array-like
        Sample values of the auxiliary variable; xbar must be non-zero.
    X : float, optional
        Known population TOTAL of x.  Given, the population total of y
        is estimated as Rhat * X.
    N : float
        Population size, for the finite population correction.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``ratio``, ``se_ratio``, ``ci_lower``, ``ci_upper``,
        ``total`` (nan if X is None), ``se_total``, ``residual_var``,
        ``fpc``, ``n``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 6, which
    gives Rhat = ybar/xbar and the large-sample variance
    (1 - f)/(n xbar^2) sum (y_i - R x_i)^2/(n - 1), and notes the
    small-sample bias of both the estimator and this variance formula.
    Chapter 6 was NOT in the scanned excerpt available to this batch,
    so the standard published form is used; the finite-population
    factor matches the ``samplingbook`` 1.2.4 convention (N - n)/N used
    throughout the sibling Cochran modules.
    """
    y = C.vec(y)
    x = C.vec(x)
    n = len(y)
    if len(x) != n:
        raise ValueError("y and x must have the same length")
    if n < 2:
        raise ValueError("a variance needs at least two observations")
    xb = sum(x) / n
    if xb == 0:
        raise ValueError("the auxiliary mean xbar must be non-zero")
    yb = sum(y) / n
    R = yb / xb
    d = [y[i] - R * x[i] for i in range(n)]
    sd2 = sum(v * v for v in d) / (n - 1)
    N = float(N)
    k = 1.0 if math.isinf(N) else (N - n) / N
    vR = k * sd2 / (n * xb * xb)
    seR = math.sqrt(vR)
    z = C.qnorm((1.0 + float(level)) / 2.0)
    if X is None:
        tot = float("nan")
        seT = float("nan")
    else:
        tot = R * float(X)
        seT = abs(float(X)) * seR
    return RichResult(payload={
        "ratio": R, "se_ratio": seR, "ci_lower": R - z * seR,
        "ci_upper": R + z * seR, "total": tot, "se_total": seT,
        "residual_var": sd2, "fpc": k, "n": n,
        "method": "Ratio estimator, Cochran Chapter 6"})


survey_ratio = ratioest


def cheatsheet():
    return "smltrt: Rhat = ybar/xbar; v = (1-f) sum(y-Rx)^2/((n-1) n xbar^2)"
