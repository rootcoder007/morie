# morie.fn -- function file (rootcoder007/morie)
"""Linear basis expansion fitted by least squares.

Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer, Section 2.6.1 and Section 2.8.3, book
pp. 30 and 35-36 (PDF pp. 49, 54-55):

    f_theta(x) = sum_{k=1..K} h_k(x) theta_k                        (2.30)
    RSS(theta) = sum_i (y_i - f_theta(x_i))^2                       (2.32)
    f_theta(x) = sum_{m=1..M} theta_m h_m(x)                        (2.43)

(2.30) and (2.43) are the same expansion, and (2.32) is the criterion
minimised.  The dictionaries offered are the ones the book names on
pp. 30 and 36: polynomial (h_k = x^k), trigonometric (cos/sin pairs),
and the linear spline basis b1(x) = 1, b2(x) = x,
b_{m+2}(x) = (x - t_m)_+ with knots t_m.

Because the basis functions carry no hidden parameters, the book notes
the minimisation has a closed form; it is solved here by the normal
equations, not by search.
"""

from __future__ import annotations

import math

from . import _s03core as k

from ._richresult import RichResult

__all__ = ["basisexp"]


def _design(x, kind, M, knots):
    n = len(x)
    if kind == "poly":
        return [[x[i] ** j for j in range(M + 1)] for i in range(n)]
    if kind == "trig":
        H = []
        for i in range(n):
            row = [1.0]
            for j in range(1, M + 1):
                row.append(math.cos(j * x[i]))
                row.append(math.sin(j * x[i]))
            H.append(row)
        return H
    if kind == "spline":
        t = list(knots)
        return [[1.0, x[i]] + [max(x[i] - tm, 0.0) for tm in t] for i in range(n)]
    raise ValueError("basisexp: kind must be 'poly', 'trig' or 'spline'")


def basisexp(x, y, kind="poly", M=3, knots=None):
    """Fit the expansion of equations (2.30)/(2.43) by least squares (2.32).

    Parameters
    ----------
    x : array-like
        N-vector of scalar inputs.
    y : array-like
        N-vector of responses.
    kind : {'poly', 'trig', 'spline'}
        Dictionary of basis functions.
    M : int
        Polynomial degree, or number of harmonics for 'trig'.
    knots : array-like, optional
        Knots t_m for the linear spline basis.

    Returns
    -------
    RichResult with keys estimate, theta, basis, fitted, residuals, rss,
    tss, r2, K, n, method.
    """
    xv = k.vec(x)
    yv = k.vec(y)
    n = len(xv)
    if n == 0:
        raise ValueError("basisexp: x is empty")
    if len(yv) != n:
        raise ValueError("basisexp: x and y must have the same length")
    M = int(M)
    if M < 0:
        raise ValueError("basisexp: M must be non-negative")
    if kind == "spline":
        if knots is None:
            raise ValueError("basisexp: the spline basis needs knots")
        knots = k.vec(knots)
    H = _design(xv, kind, M, knots)
    K = len(H[0])
    if n < K:
        raise ValueError("basisexp: fewer observations than basis functions")
    theta = k.lstsq(H, yv, 0.0)
    fitted = [sum(H[i][j] * theta[j] for j in range(K)) for i in range(n)]
    resid = [yv[i] - fitted[i] for i in range(n)]
    rss = sum(r * r for r in resid)
    ybar = sum(yv) / n
    tss = sum((v - ybar) ** 2 for v in yv)
    return RichResult(
        title="Linear basis expansion, ESL eqs. (2.30)/(2.43)",
        summary_lines=[("n", n), ("K", K), ("rss", rss)],
        payload={
            "estimate": theta[0],
            "theta": theta,
            "basis": H,
            "fitted": fitted,
            "residuals": resid,
            "rss": rss,
            "tss": tss,
            "r2": 1.0 - rss / tss if tss > 0.0 else float("nan"),
            "K": K,
            "n": n,
            "method": "Hastie-Tibshirani-Friedman (2009) ESL eqs. (2.30), (2.32), (2.43)",
        },
    )


def cheatsheet():
    return "basisexp: f(x) = sum_k h_k(x) theta_k fitted by least squares, ESL (2.30)/(2.43)"
