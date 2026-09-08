# morie.fn -- slice s04 (rootcoder007/morie)
"""Fourier basis function expansion.

Book section read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 579-631], Chapter 14, Section
14.2.1 "Fourier Basis", p. 584.  The page was read as a rendered image,
not from an extracted text layer.

ERRATUM IN THE BOOK.  The typeset list on p. 584 prints

    phi_1 = 1/sqrt(P),      phi_2 = sin(w t) /sqrt(P/2),
    phi_3 = sin(w t) /sqrt(P/2),   phi_4 = cos(2w t)/sqrt(P/2),
    phi_5 = cos(2w t)/sqrt(P/2),   phi_6 = cos(3w t)/sqrt(P/2),
    phi_7 = cos(3w t)/sqrt(P/2)

-- that is, phi_2 = phi_3, phi_4 = phi_5 and phi_6 = phi_7 as printed.
Three things on that same page show the printed list is wrong and the
intended list alternates sine and cosine:

  * a set with repeated elements is linearly dependent and is not a
    basis, which is what the page is defining;
  * the page says "the graph on interval (0,8) of the first five of
    these functions with period 4 is given in Fig. 14.1", and Fig. 14.1
    plots five distinct curves, not three;
  * the R code the page gives to reproduce that figure is
    create.fourier.basis(rangeval=c(0,8), nbasis=5, period=4) from fda,
    whose basis is the alternating one.

The alternating reading is therefore implemented:

    phi_1(t)     = 1/sqrt(P)
    phi_{2h}(t)  = sin(h w t)/sqrt(P/2)
    phi_{2h+1}(t)= cos(h w t)/sqrt(P/2),   h = 1, 2, ...

"where w is related to period P by w = 2 pi / P, and in practical
applications, this is often taken as the range of t values where the
data are observed", which is the default period below.

The a_0/2 + sum a_k cos + b_k sin series named in the function docstring
is the expansion this basis serves; the matrix returned here holds the
basis functions themselves, in the chapter's normalisation.
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
