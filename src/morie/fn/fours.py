# morie.fn -- slice s04 (rootcoder007/morie)
"""Fourier basis function expansion.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 579-631], Chapter 14, Section
14.2.1 "Fourier Basis", p. 584-585.  The chapter writes the basis as

    phi_1(t) = 1/sqrt(P)
    phi_2(t) = sin(w t)  / sqrt(P/2)
    phi_3(t) = cos(w t)  / sqrt(P/2)
    phi_4(t) = sin(2 w t)/ sqrt(P/2)
    phi_5(t) = cos(2 w t)/ sqrt(P/2)   ...

"where w is related to period P by w = 2 pi / P, and in practical
applications, this is often taken as the range of t values where the
data are observed".  The default period below is therefore max(t)-min(t).

(The typeset list in the volume prints phi_2 and phi_3 both as sin(w t)
and phi_4 and phi_5 both as cos(2 w t); the pattern of the list and the
fda basis the chapter's R code calls, create.fourier.basis, make it a
sine/cosine alternation, and Fig. 14.1 plots five distinct functions.
The alternating reading is used.)

The a_0/2 + sum a_k cos + b_k sin series named in the function docstring
is the expansion this basis serves; the matrix returned here holds the
basis functions themselves, in the chapter's normalisation, which is
what an expansion coefficient is fitted against.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["fourier_basis"]


def fourier_basis(t, n_harmonics, period=None):
    """The Chapter 14 Fourier basis evaluated at t.

    Parameters
    ----------
    t : array-like
        The points at which to evaluate the basis.
    n_harmonics : int
        Number of sine/cosine pairs; the basis has 2*n_harmonics + 1
        functions.
    period : float, optional
        P.  Defaults to the range of t, as Section 14.2.1 suggests.

    Returns
    -------
    estimate : F[0][0], the constant basis function
    F        : len(t)-by-(2*n_harmonics+1) matrix of basis values
    omega    : 2*pi/P
    period   : P
    """
    tt = k.vec(t)
    n = len(tt)
    if n == 0:
        raise ValueError("fourier_basis: t is empty")
    H = int(n_harmonics)
    if H < 0:
        raise ValueError("fourier_basis: n_harmonics must be non-negative")
    if period is None:
        P = max(tt) - min(tt)
    else:
        P = float(period)
    if P <= 0.0:
        raise ValueError("fourier_basis: the period must be positive")
    w = 2.0 * math.pi / P
    c0 = 1.0 / math.sqrt(P)
    ck = 1.0 / math.sqrt(P / 2.0)
    F = []
    for i in range(n):
        row = [c0]
        for h in range(1, H + 1):
            row.append(ck * math.sin(h * w * tt[i]))
            row.append(ck * math.cos(h * w * tt[i]))
        F.append(row)
    return RichResult(
        title="Fourier basis",
        summary_lines=[("points", n), ("harmonics", H), ("functions", 2 * H + 1)],
        payload={
            "estimate": F[0][0],
            "F": F,
            "omega": w,
            "period": P,
            "n": n,
            "method": "Chapter 14 Sect. 14.2.1 Fourier basis, phi_1 = 1/sqrt(P), sin/cos pairs scaled by 1/sqrt(P/2)",
        },
    )


def cheatsheet():
    return "fours: Fourier basis function expansion"
