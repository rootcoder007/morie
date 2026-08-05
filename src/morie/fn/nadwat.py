# morie.fn -- function file (rootcoder007/morie)
"""Nadaraya-Watson and local linear kernel regression.

Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer, Section 2.8.2, book p. 35 (PDF p. 54):

    K_lambda(x0, x) = (1/lambda) exp(-||x - x0||^2 / (2 lambda))    (2.40)
    fhat(x0) = sum_i K_lambda(x0, x_i) y_i
               / sum_i K_lambda(x0, x_i)                            (2.41)
    RSS(f_theta, x0) = sum_i K_lambda(x0, x_i)(y_i - f_theta(x_i))^2 (2.42)

with the two cases the book lists after (2.42): f_theta(x) = theta0,
which reproduces (2.41), and f_theta(x) = theta0 + theta1 x, "the
popular local linear regression model".  Note that in (2.40) lambda is
the VARIANCE of the Gaussian, not its standard deviation -- the book
says so explicitly on p. 35 -- so the exponent divides by 2 lambda.

The leading 1/lambda of (2.40) cancels in the ratio (2.41) and in the
weighted least squares (2.42); it is nonetheless kept in the returned
weights so they are the book's K_lambda and not a rescaling of it.
"""

from __future__ import annotations

import math

from . import _s03core as k

from ._richresult import RichResult

__all__ = ["nadwat"]


def nadwat(x, y, x0, lam, degree=0):
    """Equations (2.40)-(2.42).

    Parameters
    ----------
    x : array-like
        N-by-p matrix of inputs (a plain vector is one column).
    y : array-like
        N-vector of responses.
    x0 : array-like or float
        The target point.
    lam : float
        lambda of (2.40), the Gaussian variance.
    degree : {0, 1}
        0 gives the Nadaraya-Watson average (2.41); 1 gives local linear
        regression, the second bullet after (2.42).

    Returns
    -------
    RichResult with keys estimate, fit, weights, wsum, coef, lam, n, p,
    degree, method.
    """
    Xm = k.mat(x)
    yv = k.vec(y)
    n = k.nrow(Xm)
    if n == 0:
        raise ValueError("nadwat: x is empty")
    if len(yv) != n:
        raise ValueError("nadwat: x and y must have the same number of rows")
    p = k.ncol(Xm)
    z = k.vec(x0)
    if len(z) != p:
        raise ValueError("nadwat: x0 must have one entry per column of x")
    lam = float(lam)
    if lam <= 0.0:
        raise ValueError("nadwat: lambda must be positive")
    degree = int(degree)
    if degree not in (0, 1):
        raise ValueError("nadwat: degree must be 0 or 1")
    w = []
    for i in range(n):
        d2 = sum((Xm[i][a] - z[a]) ** 2 for a in range(p))
        w.append(math.exp(-d2 / (2.0 * lam)) / lam)
    wsum = sum(w)
    if wsum <= 0.0:
        raise ValueError("nadwat: all kernel weights underflowed to zero")
    if degree == 0:
        fit = sum(w[i] * yv[i] for i in range(n)) / wsum
        coef = [fit]
    else:
        q = p + 1
        if n < q:
            raise ValueError("nadwat: fewer observations than local linear coefficients")
        # weighted least squares of (2.42) on the centred design, so that
        # theta0 is the fit at x0 itself.
        D = [[1.0] + [Xm[i][a] - z[a] for a in range(p)] for i in range(n)]
        A = [[sum(w[i] * D[i][a] * D[i][b] for i in range(n)) for b in range(q)] for a in range(q)]
        rhs = [sum(w[i] * D[i][a] * yv[i] for i in range(n)) for a in range(q)]
        coef = k.ridgesolve(A, rhs, 0.0)
        fit = coef[0]
    return RichResult(
        title="Kernel regression, ESL eqs. (2.40)-(2.42)",
        summary_lines=[("n", n), ("lambda", lam), ("fit", fit)],
        payload={
            "estimate": fit,
            "fit": fit,
            "weights": w,
            "wsum": wsum,
            "coef": coef,
            "lam": lam,
            "n": n,
            "p": p,
            "degree": degree,
            "method": "Hastie-Tibshirani-Friedman (2009) ESL eqs. (2.40)-(2.42)",
        },
    )


def cheatsheet():
    return "nadwat: Nadaraya-Watson / local linear kernel regression, ESL (2.41)-(2.42)"
