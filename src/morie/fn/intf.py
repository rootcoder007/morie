# morie.fn -- function file (rootcoder007/morie)
"""Integral of a basis expansion.

Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
Springer: for x(t) = sum_k c_k phi_k(t) the integral is linear in the
coefficients,

    integral x(t) dt = sum_k c_k integral phi_k(t) dt,

so integrating a fitted curve reduces to integrating each basis
function once.  That is the formula named in the stub docstring.

The basis integrals are taken by the composite trapezoid rule over the
WHOLE grid, end intervals included.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["integrate_function"]


def _trapz(t, v):
    s = 0.0
    for i in range(len(t) - 1):
        s += 0.5 * (v[i] + v[i + 1]) * (t[i + 1] - t[i])
    return s


def integrate_function(coef, basis, t=None):
    """integral of sum_k coef[k] basis[:, k].

    Parameters
    ----------
    coef : array-like
        K coefficients.
    basis : array-like
        n-by-K matrix of basis functions evaluated on the grid.
    t : array-like, optional
        The grid.  Defaults to an equally spaced grid on [0, 1].

    Returns
    -------
    estimate : the integral of the expansion
    basis_integrals : the K individual basis integrals
    """
    c = k.vec(coef)
    B = k.mat(basis)
    n = k.nrow(B)
    K = k.ncol(B)
    if n < 2:
        raise ValueError("integrate_function: need at least two grid points")
    if len(c) != K:
        raise ValueError("integrate_function: coef must have one entry per basis column")
    if t is None:
        tt = [i / float(n - 1) for i in range(n)]
    else:
        tt = k.vec(t)
        if len(tt) != n:
            raise ValueError("integrate_function: t must match the number of basis rows")
    ints = []
    for j in range(K):
        ints.append(_trapz(tt, [B[i][j] for i in range(n)]))
    total = 0.0
    for j in range(K):
        total += c[j] * ints[j]
    return RichResult(
        title="Integral of a basis expansion",
        summary_lines=[("grid points", n), ("basis functions", K), ("integral", total)],
        payload={
            "estimate": total,
            "basis_integrals": ints,
            "n": n,
            "nbasis": K,
            "method": "Ramsay-Silverman (2005) linearity of the integral over a basis expansion, trapezoid over the whole grid",
        },
    )


def cheatsheet():
    return "intf: integral of a basis expansion"
