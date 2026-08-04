# morie.fn -- slice s04 (rootcoder007/morie)
"""Roughness penalty (integrated squared second derivative).

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 579-631], Chapter 14, Section
14.4, equations (14.10) and (14.11).  The penalised sum of squares is

    SSE_lambda(beta) = sum_i (y_i - mu - sum_l x_il beta_l)^2
                       + lambda * J_beta                        (14.10)

and "often the penalty term J_beta is based on the integrated pth order
derivatives",

    J_beta = integral_0^T (d^p/dt^p beta(t))^2 dt               (14.11)

"With the representation (14.2) of beta(t), J_beta can be expressed as
J_beta = beta' P beta, where P is a square matrix with entries
P_ij = integral_0^T phi_i^(p)(t) phi_j^(p)(t), i,j = 1,...,L1".  The
chapter says "typical chosen values of p are 1 and 2"; p = 2 is the
default here, which is the case the function's own docstring names.

The derivatives are taken by the second central difference on the
equally spaced grid and integrated by the trapezoid rule; both are exact
for the quadratics the anchor uses, and the penalty of an affine
function is exactly zero, as equation (14.11) requires.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["roughness_penalty"]


def _mat_or_col(basis):
    rows = list(basis)
    if not rows:
        return []
    if isinstance(rows[0], (list, tuple)):
        return [[float(e) for e in r] for r in rows]
    return [[float(e)] for e in rows]


def roughness_penalty(basis, lam, a=0.0, b=1.0, p=2):
    """Integrated squared derivative penalty of a basis, on a grid.

    Parameters
    ----------
    basis : array-like
        Either a vector of function values on an equally spaced grid, or
        an m-by-L matrix whose columns are basis functions on that grid.
    lam : float
        The smoothing parameter lambda of equation (14.10).
    a, b : float
        End points of the grid; the default unit interval.
    p : int
        Derivative order of equation (14.11); 1 or 2.

    Returns
    -------
    estimate : the penalty lambda * J
    penalty  : the same value
    J        : the unpenalised integral, c'Pc with c a vector of ones
    P        : the L-by-L penalty matrix of equation (14.11)
    """
    B = _mat_or_col(basis)
    m = len(B)
    if m < 3:
        raise ValueError("roughness_penalty: need at least three grid points")
    L = len(B[0])
    for r in B:
        if len(r) != L:
            raise ValueError("roughness_penalty: basis rows have unequal lengths")
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("roughness_penalty: lambda must be non-negative")
    pp = int(p)
    if pp not in (1, 2):
        raise ValueError("roughness_penalty: p must be 1 or 2")
    a = float(a)
    b = float(b)
    if not b > a:
        raise ValueError("roughness_penalty: the grid must have positive width")
    h = (b - a) / (m - 1)
    # derivative of order p at each interior grid point, central differences
    D = []
    for i in range(1, m - 1):
        row = []
        for j in range(L):
            if pp == 1:
                row.append((B[i + 1][j] - B[i - 1][j]) / (2.0 * h))
            else:
                row.append((B[i + 1][j] - 2.0 * B[i][j] + B[i - 1][j]) / (h * h))
        D.append(row)
    # P_ij = integral phi_i^(p) phi_j^(p) dt, trapezoid over the interior grid
    nD = len(D)
    P = [[0.0] * L for _ in range(L)]
    for i in range(L):
        for j in range(L):
            s = 0.0
            for r in range(nD):
                wgt = 0.5 if (r == 0 or r == nD - 1) else 1.0
                s += wgt * D[r][i] * D[r][j]
            P[i][j] = s * h
    J = 0.0
    for i in range(L):
        for j in range(L):
            J += P[i][j]
    return RichResult(
        title="Roughness penalty",
        summary_lines=[("grid", m), ("functions", L), ("order", pp)],
        payload={
            "estimate": lam * J,
            "penalty": lam * J,
            "J": J,
            "P": P,
            "n": m,
            "method": "J = integral (D^p f)^2 dt = c'Pc, Chapter 14 eqs. (14.10)-(14.11)",
        },
    )


def cheatsheet():
    return "rpnlt: Roughness penalty (integrated squared second derivative) for functional smoothing"
