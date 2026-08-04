# morie.fn -- function file (rootcoder007/morie)
"""Function-on-function linear regression.

Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
Springer, Chapter 16 "Functional linear models for functional
responses": the full functional linear model is

    y_i(s) = integral beta(s, t) x_i(t) dt + eps_i(s).

Expand the bivariate coefficient surface in a product of two bases,
beta(s, t) = sum_j sum_k B[j, k] phi_j(t) psi_k(s), and expand each
predictor curve on the t-basis and each response curve on the s-basis.
Writing Cx for the N-by-Kx matrix of predictor coefficients and Cy for
the N-by-Ky matrix of response coefficients, the model reduces to the
multivariate regression

    Cy = Cx B,      B = (Cx'Cx)^{-1} Cx'Cy,

which is the finite-dimensional form the chapter works in.  The curve
coefficients are the least-squares expansions of Section 4.2.

The reduction has an exact consequence used as this module's anchor:
if the responses ARE the predictors and the two bases are the same,
then Cy = Cx and B is the identity, whatever the data.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["function_on_function"]


def function_on_function(X, Y, basis_X, basis_Y):
    """Regress the response curves Y on the predictor curves X.

    Parameters
    ----------
    X : array-like
        N-by-Tx matrix of predictor curves, one per row.
    Y : array-like
        N-by-Ty matrix of response curves, one per row.
    basis_X : array-like
        Tx-by-Kx basis for the predictor argument t.
    basis_Y : array-like
        Ty-by-Ky basis for the response argument s.

    Returns
    -------
    estimate : B[0][0]
    B        : the Kx-by-Ky coefficient matrix of beta(s, t)
    Cx, Cy   : the predictor and response coefficient matrices
    fitted_coef : Cx B
    sse      : residual sum of squares in the response coefficients
    """
    Xm = k.mat(X)
    Ym = k.mat(Y)
    BX = k.mat(basis_X)
    BY = k.mat(basis_Y)
    N = k.nrow(Xm)
    if N == 0:
        raise ValueError("function_on_function: X is empty")
    if k.nrow(Ym) != N:
        raise ValueError("function_on_function: X and Y must have the same number of curves")
    Tx, Ty = k.ncol(Xm), k.ncol(Ym)
    if k.nrow(BX) != Tx:
        raise ValueError("function_on_function: basis_X must have one row per predictor argument")
    if k.nrow(BY) != Ty:
        raise ValueError("function_on_function: basis_Y must have one row per response argument")
    Kx, Ky = k.ncol(BX), k.ncol(BY)
    if Kx == 0 or Ky == 0:
        raise ValueError("function_on_function: a basis has no columns")
    if N < Kx:
        raise ValueError("function_on_function: need at least as many curves as predictor basis functions")
    Cx = [k.lstsq(BX, [Xm[i][p] for p in range(Tx)], 0.0) for i in range(N)]
    Cy = [k.lstsq(BY, [Ym[i][p] for p in range(Ty)], 0.0) for i in range(N)]
    cols = []
    for j in range(Ky):
        cols.append(k.lstsq(Cx, [Cy[i][j] for i in range(N)], 0.0))
    Bm = [[cols[j][i] for j in range(Ky)] for i in range(Kx)]
    fitc = k.matmul(Cx, Bm)
    sse = 0.0
    for i in range(N):
        for j in range(Ky):
            r = Cy[i][j] - fitc[i][j]
            sse += r * r
    return RichResult(
        title="Function-on-function regression",
        summary_lines=[("curves", N), ("predictor basis", Kx), ("response basis", Ky), ("SSE", sse)],
        payload={
            "estimate": Bm[0][0],
            "B": Bm,
            "Cx": Cx,
            "Cy": Cy,
            "fitted_coef": fitc,
            "sse": sse,
            "n": N,
            "method": "Ramsay-Silverman (2005) Ch.16 function-on-function regression, product-basis reduction Cy = Cx B",
        },
    )


def cheatsheet():
    return "fnlm: function-on-function linear regression"
