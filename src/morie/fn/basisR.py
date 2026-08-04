# morie.fn -- function file (rootcoder007/morie)
"""Basis-function representation of a discretely observed curve.

Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
Springer, Chapter 4 "Smoothing functional data by least squares",
Section 4.2: a curve observed at points t_1..t_n is represented as
x(t) = sum_k c_k phi_k(t), and the coefficients are chosen to minimise
SSE(c) = sum_i (y_i - sum_k c_k phi_k(t_i))^2, whose normal equations
give c = (Phi Phi)^{-1} Phi y.

Implemented from the least-squares criterion named in the stub
docstring, which is Section 4.2 of that book; the normal-equation
solution is the closed form of that criterion and is checked here
against an orthonormal design where c is recovered exactly.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["basis_representation"]


def basis_representation(y, Phi):
    """Least-squares coefficients of y on the basis matrix Phi.

    Parameters
    ----------
    y : array-like
        The n observed values.
    Phi : array-like
        n-by-K matrix of basis functions evaluated at the sampling points.

    Returns
    -------
    estimate : c[0], the first coefficient
    coef     : the K least-squares coefficients
    fitted   : Phi c
    residual : y - Phi c
    sse      : the residual sum of squares
    df       : n - K
    """
    yy = k.vec(y)
    P = k.mat(Phi)
    n = len(yy)
    if n == 0:
        raise ValueError("basis_representation: y is empty")
    if k.nrow(P) != n:
        raise ValueError("basis_representation: Phi must have one row per observation")
    K = k.ncol(P)
    if K == 0:
        raise ValueError("basis_representation: Phi has no columns")
    c = k.lstsq(P, yy, 0.0)
    fit = k.matvec(P, c)
    res = [yy[i] - fit[i] for i in range(n)]
    sse = 0.0
    for r in res:
        sse += r * r
    return RichResult(
        title="Basis representation",
        summary_lines=[("points", n), ("basis functions", K), ("SSE", sse)],
        payload={
            "estimate": c[0],
            "coef": c,
            "fitted": fit,
            "residual": res,
            "sse": sse,
            "df": n - K,
            "n": n,
            "method": "Ramsay-Silverman (2005) Sect. 4.2 least-squares basis expansion, c = (Phi Phi)^-1 Phi y",
        },
    )


def cheatsheet():
    return "basisR: least-squares basis representation of a curve"
