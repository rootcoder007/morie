# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Inverse ILR: back from coordinates to a closed composition.

Source: Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G. and
Barcelo-Vidal, C. (2003), "Isometric logratio transformations for
compositional data analysis", Mathematical Geology 35(3), 279-300,
doi:10.1023/a:1023818214614 (citation verified against Crossref).  The
default basis is the sequential binary partition printed as equation
(11) of Mateu-Figueras, Pawlowsky-Glahn and Egozcue, "The normal
distribution in some constrained sample spaces", p. 10, read as a
rendered page image.

Because the basis is orthonormal in the Aitchison inner product, the
inverse is the perturbation-linear combination

    x = C( exp( V y ) ),

with C the closure to a unit total.  ilr(ilr^-1(y)) = y exactly, which
is the round-trip anchor.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["aitchison_ilr_inverse"]


def default_sbp_basis(D):
    """Contrast matrix V (D by D-1) of the Egozcue et al. (2003) basis."""
    if D < 2:
        raise ValueError("aitchison_ilr_inverse: a composition needs at least 2 parts")
    V = [[0.0] * (D - 1) for _ in range(D)]
    for i in range(1, D):
        c = math.sqrt(i / (i + 1.0))
        for j in range(i):
            V[j][i - 1] = c / i
        V[i][i - 1] = -c
    return V


def aitchison_ilr_inverse(y, V=None, kappa=1.0):
    """Composition whose ilr coordinates are y.

    Parameters
    ----------
    y : array-like
        D-1 coordinates.
    V : sequence of sequences, optional
        D-by-(D-1) contrast matrix; defaults to the Egozcue et al. (2003)
        sequential binary partition.
    kappa : float, default 1.0
        Constant sum the result is closed to.

    Returns
    -------
    x : the closed composition
    logx_unclosed : V y, before exponentiating and closing
    """
    yy = [float(v) for v in k.vec(y)]
    if not yy:
        raise ValueError("aitchison_ilr_inverse: y is empty")
    if not (float(kappa) > 0.0):
        raise ValueError("aitchison_ilr_inverse: kappa must be positive")
    Vm = default_sbp_basis(len(yy) + 1) if V is None else [[float(a) for a in r] for r in V]
    D = len(Vm)
    p = len(Vm[0])
    if p != len(yy):
        raise ValueError("aitchison_ilr_inverse: V has %d columns but y has %d entries" % (p, len(yy)))
    lx = []
    for j in range(D):
        s = 0.0
        for i in range(p):
            s += Vm[j][i] * yy[i]
        lx.append(s)
    # subtract the max before exponentiating; closure makes the shift vanish
    m = max(lx)
    e = [math.exp(v - m) for v in lx]
    tot = 0.0
    for v in e:
        tot += v
    x = [float(kappa) * v / tot for v in e]
    return RichResult(
        title="Inverse isometric log-ratio",
        summary_lines=[("D", D)],
        payload={
            "x": x,
            "estimate": x[0],
            "logx_unclosed": lx,
            "total": float(kappa),
            "D": D,
            "method": "x = C(exp(V y)), V the Egozcue et al. (2003) SBP basis",
        },
    )


def cheatsheet():
    return "aitilri: Inverse ILR back to a closed composition"


# compact alias per ledger/NAMING.md
aitchisonilrinverse = aitchison_ilr_inverse
