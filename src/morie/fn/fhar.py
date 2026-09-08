# morie.fn -- function file (rootcoder007/morie)
"""Fourier basis, Ramsay-Silverman normalisation.

Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
Springer, Chapter 3, Section 3.3.1 "The Fourier basis system for
periodic data": the basis is

    phi_0(t)      = 1
    phi_{2r-1}(t) = sin(r omega t)
    phi_{2r}(t)   = cos(r omega t),      omega = 2 pi / P

with P the period, taken here as the range of t when not supplied.
This is the UNNORMALISED form of Section 3.3.1; the 1/sqrt(P) and
sqrt(2/P) scaled variant lives in module fours, which follows the
Montesinos Lopez normalisation instead.  The two must not be confused:
they differ by column scaling only, but the scaling changes every
coefficient.

Implemented from the sine/cosine definition named in the stub
docstring.  Verified against the closed-form values sin(0)=0, cos(0)=1
and sin(pi/2)=1, cos(pi/2)=0, and against the exact orthogonality
integral over a full period, neither of which runs through this code.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["fourier_basis"]


def fourier_basis(t, K, period=None):
    """The Section 3.3.1 Fourier basis evaluated at t.

    Parameters
    ----------
    t : array-like
        Evaluation points.
    K : int
        Number of sine/cosine pairs; the basis has 2*K + 1 functions.
    period : float, optional
        P.  Defaults to the range of t.

    Returns
    -------
    estimate : Phi[0][0], which is the constant function, always 1
    Phi      : len(t)-by-(2K+1) matrix
    omega    : 2 pi / P
    period   : P
    """
    tt = k.vec(t)
    n = len(tt)
    if n == 0:
        raise ValueError("fourier_basis: t is empty")
    KK = int(K)
    if KK < 0:
        raise ValueError("fourier_basis: K must be non-negative")
    if period is None:
        P = max(tt) - min(tt)
    else:
        P = float(period)
    if P <= 0.0:
        raise ValueError("fourier_basis: the period must be positive")
    w = 2.0 * math.pi / P
    Phi = []
    for i in range(n):
        row = [1.0]
        for r in range(1, KK + 1):
            row.append(math.sin(r * w * tt[i]))
            row.append(math.cos(r * w * tt[i]))
        Phi.append(row)
    return RichResult(
        title="Fourier basis",
        summary_lines=[("points", n), ("pairs", KK), ("functions", 2 * KK + 1)],
        payload={
            "estimate": Phi[0][0],
            "Phi": Phi,
            "omega": w,
            "period": P,
            "n": n,
            "nbasis": 2 * KK + 1,
            "method": "Ramsay-Silverman (2005) Sect. 3.3.1 Fourier basis, 1, sin(r w t), cos(r w t)",
        },
    )


def cheatsheet():
    return "fhar: Fourier basis, 1 / sin(r w t) / cos(r w t)"


# compact alias per ledger/NAMING.md
fourierbasis = fourier_basis
