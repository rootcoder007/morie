# morie.fn -- function file (rootcoder007/morie)
"""Transformation model with parametric T and nonparametric F, by GMM.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer, Section 6.1, pages 190-193 (volume
[Pages 189-232], read as rendered page images).  The model is

    T(Y, alpha) = X beta + U,   U independent of X                (6.2)

with T known up to the finite-dimensional alpha and F_U left
unrestricted.  Page 192 explains why nonlinear least squares is
inconsistent here -- dT(Y, alpha)/d alpha is correlated with U, so it is
not a valid instrument -- and replaces it with a vector of valid
instruments W satisfying E(WU) = 0 and dim(W) >= dim(beta) + 1.  The
estimator solves

    minimize over (a, b):  G_n(a, b) Omega_n G_n(a, b),           (6.8)
    G_n(a, b) = n^-1 sum_i W_i [T(Y_i, a) - X_i b],               (6.7)

and the book names Omega_n = (W W)^-1 as one possible choice, which
makes (6.8) the nonlinear two-stage least-squares estimator.  That
choice is used here, and the instruments are X augmented with the
squares of its non-intercept columns, which are the "powers,
cross-products, and other nonlinear functions of components of X" the
book proposes on p. 192.

Two transformation families are offered, both taken from the book:
"boxcox" is (6.3), T(y, a) = (y^a - 1)/a for a nonzero and log y at
a = 0, and "bickel-doksum" is (6.4), T(y, a) = (|y|^a sgn(y) - 1)/a.
For any fixed a the criterion is a quadratic in b with the explicit
solution b(a) = [(X W) Omega (W X)]^-1 (X W) Omega (W T(Y, a)), so only
a one-dimensional search over a remains; it is done on a fixed grid
followed by golden-section refinement, so nothing is random.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_parametric_T"]

_GR = 0.6180339887498949


def _transform(yv, a, family):
    out = []
    if family == "boxcox":
        for v in yv:
            if v <= 0.0:
                raise ValueError("horowitz_parametric_T: the Box-Cox family of (6.3) needs positive Y")
            out.append(math.log(v) if a == 0.0 else (v ** a - 1.0) / a)
    elif family == "bickel-doksum":
        if a <= 0.0:
            raise ValueError("horowitz_parametric_T: the Bickel-Doksum family of (6.4) needs a > 0")
        for v in yv:
            s = 1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0)
            out.append(((abs(v) ** a) * s - 1.0) / a)
    else:
        raise ValueError("horowitz_parametric_T: T_family must be boxcox or bickel-doksum")
    return out


def horowitz_parametric_T(x, y, T_family="boxcox", a_lo=-2.0, a_hi=2.0, ngrid=81, refine=80):
    """NL2SLS on the moment condition (6.7): alpha, beta and the criterion.

    Parameters
    ----------
    x : array-like
        n-by-p design matrix; the first column should be the intercept.
    y : array-like
        Outcomes, positive for the Box-Cox family.
    T_family : str
        "boxcox" (6.3) or "bickel-doksum" (6.4).
    a_lo, a_hi, ngrid, refine :
        The deterministic one-dimensional search over alpha.

    Returns
    -------
    theta_hat : alpha, the transformation parameter
    beta_hat  : the coefficients at that alpha
    criterion : the minimised value of (6.8)
    """
    XX = core.mat(x)
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("horowitz_parametric_T: y is empty")
    if len(XX) != n:
        raise ValueError("horowitz_parametric_T: x has a different number of rows than y")
    p = len(XX[0])
    if p < 2:
        raise ValueError("horowitz_parametric_T: need an intercept and at least one covariate")
    W = []
    for i in range(n):
        row = list(XX[i])
        for k in range(1, p):
            row.append(XX[i][k] * XX[i][k])
        W.append(row)
    q = len(W[0])
    WtW = core.crossprod(W)
    XtW = [[0.0] * q for _ in range(p)]
    for i in range(n):
        for a in range(p):
            for b in range(q):
                XtW[a][b] += XX[i][a] * W[i][b]

    def solve_b(Ty):
        WtT = [0.0] * q
        for i in range(n):
            for b in range(q):
                WtT[b] += W[i][b] * Ty[i]
        # Omega = (W'W)^-1, so form Omega %*% (W'X) and Omega %*% (W'T)
        OWX = []
        for a in range(p):
            OWX.append(core.ridgesolve(WtW, [XtW[a][b] for b in range(q)], 1e-12))
        OWT = core.ridgesolve(WtW, WtT, 1e-12)
        A = [[0.0] * p for _ in range(p)]
        rhs = [0.0] * p
        for a in range(p):
            for c in range(p):
                s = 0.0
                for b in range(q):
                    s += XtW[a][b] * OWX[c][b]
                A[a][c] = s
            s = 0.0
            for b in range(q):
                s += XtW[a][b] * OWT[b]
            rhs[a] = s
        return core.ridgesolve(A, rhs, 1e-12)

    def crit(a):
        Ty = _transform(yv, a, T_family)
        b = solve_b(Ty)
        g = [0.0] * q
        for i in range(n):
            r = Ty[i]
            for k in range(p):
                r -= XX[i][k] * b[k]
            for c in range(q):
                g[c] += W[i][c] * r
        for c in range(q):
            g[c] /= n
        og = core.ridgesolve(WtW, g, 1e-12)
        s = 0.0
        for c in range(q):
            s += g[c] * og[c]
        return s, b

    lo = float(a_lo)
    hi = float(a_hi)
    m = int(ngrid)
    if m < 3 or hi <= lo:
        raise ValueError("horowitz_parametric_T: need a_lo < a_hi and ngrid >= 3")
    best = None
    bi = 0
    for i in range(m):
        a = lo + (hi - lo) * i / (m - 1)
        v, _ = crit(a)
        if best is None or v < best:
            best = v
            bi = i
    step = (hi - lo) / (m - 1)
    left = lo + (hi - lo) * max(bi - 1, 0) / (m - 1)
    right = lo + (hi - lo) * min(bi + 1, m - 1) / (m - 1)
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
    Ty = _transform(yv, a_hat, T_family)
    resid = []
    for i in range(n):
        r = Ty[i]
        for k in range(p):
            r -= XX[i][k] * b_hat[k]
        resid.append(r)
    return RichResult(
        title="Transformation model, parametric T and nonparametric F",
        summary_lines=[("n", n), ("alpha", a_hat)],
        payload={
            "estimate": a_hat,
            "theta_hat": a_hat,
            "alpha_hat": a_hat,
            "beta_hat": b_hat,
            "criterion": val,
            "resid": resid,
            "T_family": T_family,
            "n": n,
            "method": "Horowitz (2009) eq. (6.7)-(6.8) NL2SLS with Omega = (W'W)^-1",
        },
    )


def cheatsheet():
    return "hrztpar: Transformation model with parametric T, nonparametric F"
