# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Isometric log-ratio (ILR) transform via a sequential binary partition.

Source consulted (rendered page image, not the text layer):
Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., "The normal
distribution in some constrained sample spaces", pp. 10-11, which
reproduces the Aitchison inner product

    <x, x*>_a = (1/D) sum_{i<j} ln(x_i/x_j) ln(x*_i/x*_j)            (10)

and the default orthonormal basis of Egozcue et al. (2003), whose
coordinates are

    y_i = 1/sqrt(i(i+1)) * ln( (x_1 x_2 ... x_i) / x_{i+1}^i ),
                                                 i = 1, ..., D-1     (11)

Primary reference verified against Crossref: Egozcue, J. J.,
Pawlowsky-Glahn, V., Mateu-Figueras, G. and Barcelo-Vidal, C. (2003),
"Isometric logratio transformations for compositional data analysis",
Mathematical Geology 35(3), 279-300, doi:10.1023/a:1023818214614.

Equation (11) is algebraically the contrast form used here,

    y_i = sqrt(i/(i+1)) * ( (1/i) sum_{j<=i} ln x_j  -  ln x_{i+1} )
        = v_i' clr(x),

with clr(x) = ln x - mean(ln x); the mean subtraction cancels because
every contrast column sums to zero, so the two routes agree exactly.
That identity is the module's first anchor.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["aitchison_ilr"]


def clr(x):
    """Centred log-ratio: ln x_i - mean_j ln x_j."""
    lg = [math.log(v) for v in x]
    m = sum(lg) / len(lg)
    return [v - m for v in lg]


def default_sbp_basis(D):
    """Contrast matrix V (D rows, D-1 columns) of the Egozcue (2003) basis.

    Column i (1-based) holds +sqrt(i/(i+1))/i on rows 1..i, -sqrt(i/(i+1))
    on row i+1, zero elsewhere.  Every column sums to zero and has unit
    Euclidean norm, so V'V = I_{D-1}.
    """
    if D < 2:
        raise ValueError("aitchison_ilr: a composition needs at least 2 parts")
    V = [[0.0] * (D - 1) for _ in range(D)]
    for i in range(1, D):
        c = math.sqrt(i / (i + 1.0))
        for j in range(i):
            V[j][i - 1] = c / i
        V[i][i - 1] = -c
    return V


def _check_simplex(x):
    if len(x) < 2:
        raise ValueError("aitchison_ilr: a composition needs at least 2 parts")
    for v in x:
        if not (v > 0.0):
            raise ValueError("aitchison_ilr: every part must be strictly positive")


def aitchison_ilr(x, V=None):
    """ilr coordinates of a composition.

    Parameters
    ----------
    x : array-like
        A D-part composition with strictly positive entries.  It need not
        be closed; the transform is scale invariant.
    V : sequence of sequences, optional
        D-by-(D-1) contrast matrix whose columns are the clr coefficients
        of an orthonormal basis.  Defaults to the Egozcue et al. (2003)
        sequential binary partition of equation (11).

    Returns
    -------
    y : the D-1 ilr coordinates
    norm : the Aitchison norm ||x||_a, which equals the Euclidean norm
        of y because ilr is an isometry
    """
    xx = [float(v) for v in k.vec(x)]
    _check_simplex(xx)
    D = len(xx)
    Vm = default_sbp_basis(D) if V is None else [[float(a) for a in r] for r in V]
    if len(Vm) != D:
        raise ValueError("aitchison_ilr: V has the wrong number of rows")
    p = len(Vm[0])
    for r in Vm:
        if len(r) != p:
            raise ValueError("aitchison_ilr: V is ragged")
    z = clr(xx)
    y = []
    for i in range(p):
        s = 0.0
        for j in range(D):
            s += Vm[j][i] * z[j]
        y.append(s)
    nrm = 0.0
    for v in y:
        nrm += v * v
    nrm = math.sqrt(nrm)
    # Aitchison norm straight from equation (10), never touching y.
    a2 = 0.0
    lg = [math.log(v) for v in xx]
    for i in range(D):
        for j in range(i + 1, D):
            d = lg[i] - lg[j]
            a2 += d * d
    a2 = a2 / D
    return RichResult(
        title="Isometric log-ratio coordinates",
        summary_lines=[("D", D), ("norm", nrm)],
        payload={
            "y": y,
            "estimate": y[0] if y else float("nan"),
            "clr": z,
            "norm": nrm,
            "aitchison_norm": math.sqrt(a2),
            "D": D,
            "method": "ilr(x) = V' clr(x), V the Egozcue et al. (2003) SBP basis, eq. (11)",
        },
    )


def cheatsheet():
    return "aitilr: Isometric log-ratio (ILR) transform via SBP"


# compact alias per ledger/NAMING.md
aitchisonilr = aitchison_ilr
