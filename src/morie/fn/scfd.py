# morie.fn -- function file (rootcoder007/morie)
"""Scalar-on-function linear regression.

Reiss and Ogden (2007), "Functional principal component regression and
functional partial least squares", *JASA* 102(479), 984-996, and the
underlying model of Ramsay and Silverman (2005), *Functional Data
Analysis*, 2nd ed., Chapter 15: a scalar response is regressed on a
functional predictor,

    y_i = alpha + integral beta(t) x_i(t) dt + eps_i.

Expanding the coefficient function in a K-term basis,
beta(t) = sum_k b_k phi_k(t), the integral becomes linear in b:

    integral beta x_i = sum_k b_k integral phi_k(t) x_i(t) dt = (J b)_i,
    J[i, k] = integral phi_k(t) x_i(t) dt,

so the model is the ordinary linear regression of y on the design
[1, J].  This basis-expansion reduction is what makes the problem
finite dimensional and is the same device Reiss and Ogden use before
choosing the basis by principal components.

The inner products J are taken by the composite trapezoid rule over the
whole observation grid.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _fdacore as fda
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["scalar_on_function"]


def scalar_on_function(X, Y, basis, t=None):
    """Regress the scalars Y on the curves X through a basis for beta.

    Parameters
    ----------
    X : array-like
        N-by-T matrix, one predictor curve per row.
    Y : array-like
        N scalar responses.
    basis : array-like
        T-by-K matrix of basis functions evaluated on the grid.
    t : array-like, optional
        The grid of length T.  Defaults to equally spaced on [0, 1].

    Returns
    -------
    estimate : alpha, the intercept
    alpha    : the intercept
    coef     : the K coefficients of beta in the basis
    beta     : beta(t) evaluated on the grid
    J        : the N-by-K matrix of inner products
    fitted, residual, sse, r2
    """
    Xm = k.mat(X)
    yy = k.vec(Y)
    B = k.mat(basis)
    N = k.nrow(Xm)
    if N == 0:
        raise ValueError("scalar_on_function: X is empty")
    if len(yy) != N:
        raise ValueError("scalar_on_function: Y must have one value per curve")
    T = k.ncol(Xm)
    if k.nrow(B) != T:
        raise ValueError("scalar_on_function: basis must have one row per argument value")
    K = k.ncol(B)
    if K == 0:
        raise ValueError("scalar_on_function: basis has no columns")
    if T < 2:
        raise ValueError("scalar_on_function: need at least two argument values")
    if N <= K:
        raise ValueError("scalar_on_function: need more curves than basis functions")
    tt = fda.grid(T) if t is None else k.vec(t)
    if len(tt) != T:
        raise ValueError("scalar_on_function: t must match the number of argument values")
    J = []
    for i in range(N):
        row = []
        for j in range(K):
            row.append(fda.trapz(tt, [B[p][j] * Xm[i][p] for p in range(T)]))
        J.append(row)
    Z = [[1.0] + J[i] for i in range(N)]
    ab = k.lstsq(Z, yy, 0.0)
    alpha = ab[0]
    b = ab[1:]
    fit = k.matvec(Z, ab)
    res = [yy[i] - fit[i] for i in range(N)]
    sse = 0.0
    for r in res:
        sse += r * r
    ybar = k.mean(yy)
    sst = 0.0
    for v in yy:
        sst += (v - ybar) * (v - ybar)
    beta = []
    for p in range(T):
        s = 0.0
        for j in range(K):
            s += b[j] * B[p][j]
        beta.append(s)
    return RichResult(
        title="Scalar-on-function regression",
        summary_lines=[("curves", N), ("basis functions", K), ("SSE", sse)],
        payload={
            "estimate": alpha,
            "alpha": alpha,
            "coef": b,
            "beta": beta,
            "J": J,
            "fitted": fit,
            "residual": res,
            "sse": sse,
            "r2": 1.0 - sse / sst if sst > 0.0 else float("nan"),
            "df": N - K - 1,
            "n": N,
            "method": "Reiss-Ogden (2007) / Ramsay-Silverman (2005) Ch.15 scalar-on-function regression by basis expansion of beta",
        },
    )


def cheatsheet():
    return "scfd: scalar-on-function linear regression"
