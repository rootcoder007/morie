# morie.fn -- function file (rootcoder007/morie)
"""Function-on-function (fully functional) linear regression.

Book pages read as rendered images (pdftoppm, 130 dpi), not from an
extracted text layer: Ramsay and Silverman (2005), *Functional Data
Analysis*, 2nd ed., Springer, Chapter 16 "Functional linear models for
functional responses", Section 16.1 p. 279 and Section 16.4.1
pp. 291-292.

The model of equation (16.6) is

    y*(t) = integral z*(s) beta(s, t) ds + eps(t),               (16.6)

with the tensor-product expansion of equation (16.3)

    beta(s, t) = theta'(s) B eta(t),                             (16.3)

theta a basis of K1 functions of s and eta a basis of K2 functions of
t.  Substituting (16.3) into (16.6) gives equation (16.7),

    y*(t) = Z* B eta(t) + eps(t),   Z* = integral z*(s) theta'(s) ds,
                                                          (16.7), (16.8)

and the unweighted criterion (16.2) is minimised by the normal
equations of equation (16.9), p. 292:

    Z*' Z* B integral eta(t) eta'(t) dt = Z*' integral y(t) eta'(t) dt.

Writing J_etaeta = integral eta(t) eta'(t) dt this is solved here as

    B = (Z*' Z*)^-1 (Z*' M) J_etaeta^-1,   M_il = integral y_i(t) eta_l(t) dt,

which is equation (16.9) rearranged, not an approximation of it.  The
Kronecker form (16.10) is the same equation vectorised and is not used;
it would give the identical B.

No regularisation is applied: this is Section 16.4.1, "Fitting the model
without regularization".  The roughness-penalised variant of Section
16.4.2, equations (16.13)-(16.15), needs the two differential operators
L_s and L_t as inputs and is not what this function's signature offers.

All integrals are composite trapezoid rules over the WHOLE grid, end
intervals included.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["function_on_function"]


def _trapz(t, v):
    """Composite trapezoid rule over the whole of t."""
    s = 0.0
    for i in range(len(t) - 1):
        s += 0.5 * (v[i] + v[i + 1]) * (t[i + 1] - t[i])
    return s


def function_on_function(X, Y, basis_X, basis_Y, s=None, t=None):
    """Fit y_i(t) = integral z_i(s) beta(s, t) ds by equation (16.9).

    Parameters
    ----------
    X : array-like
        N-by-ns matrix of covariate curves z_i(s), one curve per row.
    Y : array-like
        N-by-nt matrix of response curves y_i(t), one curve per row.
    basis_X : array-like
        ns-by-K1 matrix, the basis theta evaluated on the s grid.
    basis_Y : array-like
        nt-by-K2 matrix, the basis eta evaluated on the t grid.
    s, t : array-like, optional
        The two grids.  Default to equally spaced grids on [0, 1].

    Returns
    -------
    estimate : B[0][0]
    B        : K1-by-K2 coefficient matrix of equation (16.3)
    beta     : ns-by-nt matrix, beta(s, t) = theta'(s) B eta(t)
    Z        : the N-by-K1 matrix Z* of equation (16.8)
    fitted   : N-by-nt matrix of fitted response curves
    residual : Y - fitted
    sse      : LMSSE of equation (16.2), integrated over t and summed over i
    ssy      : the same criterion for the null fit beta = 0
    r2       : 1 - sse/ssy
    """
    Xm = k.mat(X)
    Ym = k.mat(Y)
    Th = k.mat(basis_X)
    Et = k.mat(basis_Y)
    N = k.nrow(Xm)
    if N == 0:
        raise ValueError("function_on_function: X is empty")
    if k.nrow(Ym) != N:
        raise ValueError("function_on_function: X and Y must have the same number of curves")
    ns = k.ncol(Xm)
    nt = k.ncol(Ym)
    if ns < 2 or nt < 2:
        raise ValueError("function_on_function: need at least two points on each grid")
    if k.nrow(Th) != ns:
        raise ValueError("function_on_function: basis_X must have one row per s point")
    if k.nrow(Et) != nt:
        raise ValueError("function_on_function: basis_Y must have one row per t point")
    K1 = k.ncol(Th)
    K2 = k.ncol(Et)
    if K1 == 0 or K2 == 0:
        raise ValueError("function_on_function: a basis has no columns")
    if N < K1:
        raise ValueError("function_on_function: need at least K1 curves")
    ss = [i / float(ns - 1) for i in range(ns)] if s is None else k.vec(s)
    tt = [i / float(nt - 1) for i in range(nt)] if t is None else k.vec(t)
    if len(ss) != ns:
        raise ValueError("function_on_function: s must match the columns of X")
    if len(tt) != nt:
        raise ValueError("function_on_function: t must match the columns of Y")

    # Z* of equation (16.8): Z[i][k] = integral z_i(s) theta_k(s) ds
    Z = []
    for i in range(N):
        row = []
        for c in range(K1):
            row.append(_trapz(ss, [Xm[i][a] * Th[a][c] for a in range(ns)]))
        Z.append(row)
    # M[i][l] = integral y_i(t) eta_l(t) dt, the right side of (16.9)
    M = []
    for i in range(N):
        row = []
        for c in range(K2):
            row.append(_trapz(tt, [Ym[i][b] * Et[b][c] for b in range(nt)]))
        M.append(row)
    # J_etaeta = integral eta(t) eta'(t) dt
    J = [[0.0] * K2 for _ in range(K2)]
    for c in range(K2):
        for d in range(K2):
            J[c][d] = _trapz(tt, [Et[b][c] * Et[b][d] for b in range(nt)])
    A = k.crossprod(Z)          # Z*' Z*,  K1 by K1
    R = k.matmul(k.tr(Z), M)    # Z*' M,   K1 by K2
    # solve A W = R column by column, then W = B J so B J = W
    W = []
    for c in range(K2):
        W.append(k.ridgesolve(A, [R[r][c] for r in range(K1)], 0.0))
    # W is K2 rows of length K1; transpose back to K1 by K2
    Wm = [[W[c][r] for c in range(K2)] for r in range(K1)]
    B = []
    for r in range(K1):
        B.append(k.ridgesolve(J, Wm[r], 0.0))

    beta = [[0.0] * nt for _ in range(ns)]
    for a in range(ns):
        for b in range(nt):
            v = 0.0
            for c in range(K1):
                for d in range(K2):
                    v += Th[a][c] * B[c][d] * Et[b][d]
            beta[a][b] = v
    fitted = [[0.0] * nt for _ in range(N)]
    for i in range(N):
        for b in range(nt):
            v = 0.0
            for c in range(K1):
                for d in range(K2):
                    v += Z[i][c] * B[c][d] * Et[b][d]
            fitted[i][b] = v
    resid = [[Ym[i][b] - fitted[i][b] for b in range(nt)] for i in range(N)]
    sse = 0.0
    ssy = 0.0
    for i in range(N):
        sse += _trapz(tt, [resid[i][b] * resid[i][b] for b in range(nt)])
        ssy += _trapz(tt, [Ym[i][b] * Ym[i][b] for b in range(nt)])
    r2 = 1.0 - sse / ssy if ssy > 0.0 else 0.0
    return RichResult(
        title="Function-on-function regression",
        summary_lines=[("curves", N), ("K1", K1), ("K2", K2), ("R2", r2)],
        payload={
            "estimate": B[0][0],
            "B": B,
            "beta": beta,
            "Z": Z,
            "J": J,
            "fitted": fitted,
            "residual": resid,
            "sse": sse,
            "ssy": ssy,
            "r2": r2,
            "n": N,
            "method": "Ramsay-Silverman (2005) eqs. (16.3), (16.6)-(16.9), unregularised tensor-product fit",
        },
    )


def cheatsheet():
    return "fnlm: function-on-function linear regression, beta(s,t) tensor-product fit"
