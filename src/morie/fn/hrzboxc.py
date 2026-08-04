# morie.fn -- function file (rootcoder007/morie)
"""Box-Cox regression by the minimum-distance estimator of Foster et al.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer, Section 6.1.2, page 195 (volume [Pages 189-232],
read as a rendered page image).  The model is (6.2) with T the Box-Cox
transformation (6.3), page 190,

    T(y, a) = (y^a - 1) / a   for a != 0,      log y   for a = 0,

and F_U left unrestricted.  For any candidate a the slope solves the
ordinary least-squares problem printed at the top of p. 195,

    b_n(a) = (sum_i X_i X_i')^-1 sum_i X_i T(Y_i, a),

the residuals U_hat_i = T(Y_i, a) - X_i' b_n(a) give the empirical CDF
F_n[u; a, b_n(a)] = n^-1 sum_i I(U_hat_i < u), and since
P(Y < y) = E F_U[T(y, a) - X' beta], alpha is estimated by the
minimum-distance problem of the same page,

    minimize over a:  R_n[a, b_n(a)]
      = n^-1 sum_i integral_0^inf {I(Y_i < u) - F_n[T(u, a) - X_i' b_n(a)]}^2 w(u) du.

BOOK NOTE.  The displayed criterion on p. 195 prints the inner argument
as T(y, a); y is not bound anywhere in the expression while u is the
variable of integration, and the identity P(Y < y) = E F_U[T(y,a) - X b]
two lines above fixes the reading: it is T(u, a).  That is what is
implemented.

The weight w is, as the book requires, positive, deterministic and
bounded: the uniform density on (0, max Y].  The outer integral is a
trapezoid rule on a fixed grid and the search over a is a fixed grid
followed by golden-section refinement, so nothing here is random.
Foster, Tian and Wei (2001) is the source the book credits for the
estimator and for Theorem 6.1.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_box_cox"]

_GR = 0.6180339887498949


def _bc(v, a):
    return math.log(v) if a == 0.0 else (v ** a - 1.0) / a


def horowitz_box_cox(x, y, a_lo=-2.0, a_hi=2.0, ngrid=81, refine=60, nu=201):
    """Box-Cox alpha by minimum distance, beta by least squares at that alpha.

    Parameters
    ----------
    x : array-like
        n-by-p design matrix; the first column should be the intercept.
    y : array-like
        Strictly positive outcomes.
    a_lo, a_hi, ngrid, refine :
        The deterministic one-dimensional search over alpha.
    nu : int
        Points in the trapezoid rule for the integral over u.

    Returns
    -------
    lambda_hat : alpha, the Box-Cox parameter
    beta_hat   : b_n(alpha)
    criterion  : R_n at the minimiser
    """
    XX = core.mat(x)
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("horowitz_box_cox: y is empty")
    if len(XX) != n:
        raise ValueError("horowitz_box_cox: x has a different number of rows than y")
    for v in yv:
        if v <= 0.0:
            raise ValueError("horowitz_box_cox: the Box-Cox transformation of (6.3) needs positive Y")
    p = len(XX[0])
    m = int(nu)
    if m < 3:
        raise ValueError("horowitz_box_cox: nu must be at least 3")
    umax = yv[0]
    for v in yv:
        if v > umax:
            umax = v
    ug = [umax * (k + 1) / m for k in range(m)]
    du = umax / m
    wt = 1.0 / umax

    def fit(a):
        Ty = [_bc(v, a) for v in yv]
        b = core.lstsq(XX, Ty, 1e-12)
        uh = []
        for i in range(n):
            r = Ty[i]
            for k in range(p):
                r -= XX[i][k] * b[k]
            uh.append(r)
        return Ty, b, uh

    def crit(a):
        Ty, b, uh = fit(a)
        us = sorted(uh)
        tot = 0.0
        for k in range(m):
            u = ug[k]
            tu = _bc(u, a)
            for i in range(n):
                xb = 0.0
                for j in range(p):
                    xb += XX[i][j] * b[j]
                z = tu - xb
                # F_n(z) = n^-1 sum I(U_hat < z), by binary search on the sorted residuals
                lo = 0
                hi = n
                while lo < hi:
                    mid = (lo + hi) // 2
                    if us[mid] < z:
                        lo = mid + 1
                    else:
                        hi = mid
                fn = lo / n
                d = (1.0 if yv[i] < u else 0.0) - fn
                tot += d * d * wt * du
        return tot / n, b

    lo = float(a_lo)
    hi = float(a_hi)
    g = int(ngrid)
    if g < 3 or hi <= lo:
        raise ValueError("horowitz_box_cox: need a_lo < a_hi and ngrid >= 3")
    best = None
    bi = 0
    for i in range(g):
        a = lo + (hi - lo) * i / (g - 1)
        v, _ = crit(a)
        if best is None or v < best:
            best = v
            bi = i
    step = (hi - lo) / (g - 1)
    left = lo + (hi - lo) * max(bi - 1, 0) / (g - 1)
    right = lo + (hi - lo) * min(bi + 1, g - 1) / (g - 1)
    if right - left < step:
        left = max(left - step, lo)
        right = min(right + step, hi)
    c1 = right - _GR * (right - left)
    c2 = left + _GR * (right - left)
    f1, _ = crit(c1)
    f2, _ = crit(c2)
    for _ in range(int(refine)):
        if f1 < f2:
            right = c2
            c2 = c1
            f2 = f1
            c1 = right - _GR * (right - left)
            f1, _ = crit(c1)
        else:
            left = c1
            c1 = c2
            f1 = f2
            c2 = left + _GR * (right - left)
            f2, _ = crit(c2)
    a_hat = 0.5 * (left + right)
    val, b_hat = crit(a_hat)
    Ty, b_hat, uh = fit(a_hat)
    return RichResult(
        title="Box-Cox regression by minimum distance",
        summary_lines=[("n", n), ("lambda", a_hat)],
        payload={
            "estimate": a_hat,
            "lambda_hat": a_hat,
            "beta_hat": b_hat,
            "criterion": val,
            "resid": uh,
            "n": n,
            "method": "Horowitz (2009) Sec. 6.1.2 p.195 minimum distance (Foster, Tian and Wei 2001)",
        },
    )


def cheatsheet():
    return "hrzboxc: Box-Cox regression model: T_lambda(Y) = X'beta + U"


# compact alias per ledger/NAMING.md
horowitzboxcox = horowitz_box_cox
