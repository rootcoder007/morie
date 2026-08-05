# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Kabsch optimal superposition.

Kabsch (1976), "A solution for the best rotation to relate two sets of
vectors", Acta Cryst. A32(5):922-923, doi:10.1107/S0567739476001873,
with the reflection correction of Kabsch (1978), Acta Cryst.
A34(5):827-828, doi:10.1107/S0567739478001680.  Both sets are centred,
the cross-covariance H = P'Q is formed, and with H = U S V' the
optimal rotation is

    R = V diag(1, ..., 1, d) U',   d = sign(det(V U')),

the last entry guarding against an improper rotation (a reflection),
which fits the points equally well but is not a rotation.  The SVD is
taken from the Jacobi eigendecomposition of H'H, so no external linear
algebra is used.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["kabsch_superpose"]


def _det(A):
    n = len(A)
    M = [row[:] for row in A]
    det = 1.0
    for i in range(n):
        p = i
        for r in range(i, n):
            if abs(M[r][i]) > abs(M[p][i]):
                p = r
        if abs(M[p][i]) < 1e-300:
            return 0.0
        if p != i:
            M[i], M[p] = M[p], M[i]
            det = -det
        det *= M[i][i]
        for r in range(i + 1, n):
            fac = M[r][i] / M[i][i]
            for cc in range(i, n):
                M[r][cc] -= fac * M[i][cc]
    return det


def kabsch_superpose(coords1, coords2):
    """Rotation and RMSD superposing coords1 onto coords2."""
    P = core.mat(coords1)
    Q = core.mat(coords2)
    if len(P) == 0 or len(Q) == 0:
        raise ValueError("kabsch_superpose: coordinate set is empty")
    if len(P) != len(Q):
        raise ValueError("kabsch_superpose: coordinate sets have different point counts")
    d = len(P[0])
    if len(Q[0]) != d:
        raise ValueError("kabsch_superpose: coordinate sets have different dimensions")
    n = len(P)
    cp = [sum(P[i][j] for i in range(n)) / n for j in range(d)]
    cq = [sum(Q[i][j] for i in range(n)) / n for j in range(d)]
    A = [[P[i][j] - cp[j] for j in range(d)] for i in range(n)]
    B = [[Q[i][j] - cq[j] for j in range(d)] for i in range(n)]
    H = [[sum(A[i][j] * B[i][k] for i in range(n)) for k in range(d)] for j in range(d)]
    HtH = [[sum(H[i][j] * H[i][k] for i in range(d)) for k in range(d)] for j in range(d)]
    vals, vecs = core.jacobi(HtH)
    idx = list(range(d))[::-1]
    V = [[vecs[r][idx[j]] for j in range(d)] for r in range(d)]
    sing = [math.sqrt(max(vals[idx[j]], 0.0)) for j in range(d)]
    if sing[d - 1] <= 1e-12 * max(sing[0], 1e-300):
        raise ValueError("kabsch_superpose: degenerate configuration, the cross-covariance is rank deficient")
    U = [[0.0] * d for _ in range(d)]
    for j in range(d):
        col = [sum(H[r][k] * V[k][j] for k in range(d)) / sing[j] for r in range(d)]
        for r in range(d):
            U[r][j] = col[r]
    VUt = [[sum(V[i][k] * U[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
    dd = 1.0 if _det(VUt) >= 0 else -1.0
    R = [[sum(V[i][k] * (dd if k == d - 1 else 1.0) * U[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
    ss = 0.0
    for i in range(n):
        for j in range(d):
            rp = sum(R[j][k] * A[i][k] for k in range(d))
            ss += (rp - B[i][j]) ** 2
    rmsd = math.sqrt(ss / n)
    return RichResult(
        title="Kabsch superposition",
        summary_lines=[("points", n), ("dimension", d)],
        payload={
            "estimate": rmsd,
            "rmsd": rmsd,
            "rotation": R,
            "det_rotation": _det(R),
            "singular_values": sing,
            "n": n,
            "method": "R = V diag(1,...,d) U' from H = P'Q, Kabsch (1976, 1978)",
        },
    )


def cheatsheet():
    return "kabsch: Kabsch optimal superposition"
