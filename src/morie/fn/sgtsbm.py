# morie.fn -- slice s03 (rootcoder007/morie)
"""Two-way spectral partition by the sign of the Fiedler vector.

Sources consulted: Fiedler, M. (1973).  Algebraic connectivity of
graphs.  *Czechoslovak Mathematical Journal* 23(2), 298-305, which
identifies the eigenvector of the second-smallest eigenvalue of the
Laplacian as the one that orders a graph's vertices; and Shi, J. and
Malik, J. (2000), *IEEE TPAMI* 22(8), 888-905, which shows that the
second-smallest generalised eigenvector of

    (D - W) y = lambda D y

is the real-valued relaxation of the normalised cut, and that
thresholding it gives the partition.  Neither paper was retrievable here
as a full text; both results are quoted in their standard published
form.

The eigenproblem is solved by cyclic Jacobi on the symmetric normalised
Laplacian D^(-1/2)(D - W)D^(-1/2), whose eigenvectors relate to the
generalised ones by y = D^(-1/2) z.  Eigenvector signs are not
determined by the eigenproblem, so the vector is sign-fixed (its
largest-magnitude entry is made positive) before thresholding -- without
that the two arms would return complementary partitions.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["sgt_spectral_clustering_2"]


def sgt_spectral_clustering_2(A, normalized=True):
    """Partition a weighted graph in two by the Fiedler sign.

    Returns
    -------
    RichResult with payload:
        labels    : 0/1 per node
        estimate  : the algebraic connectivity (second-smallest eigenvalue)
        fiedler   : the thresholded eigenvector
        eigenvalues : the full spectrum, ascending
    """
    W = k.mat(A)
    n = len(W)
    d = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += W[i][j]
        d.append(s)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            L[i][j] = (d[i] if i == j else 0.0) - W[i][j]
    if normalized:
        for i in range(n):
            for j in range(n):
                di = math.sqrt(d[i]) if d[i] > 0.0 else 0.0
                dj = math.sqrt(d[j]) if d[j] > 0.0 else 0.0
                L[i][j] = L[i][j] / (di * dj) if di > 0.0 and dj > 0.0 else 0.0
    vals, vecs = k.jacobi(L)
    fied = [vecs[i][1] if n > 1 else 0.0 for i in range(n)]
    if normalized:
        fied = [fied[i] / math.sqrt(d[i]) if d[i] > 0.0 else 0.0 for i in range(n)]
        big = 0
        for i in range(n):
            if abs(fied[i]) > abs(fied[big]) + 1e-15:
                big = i
        if fied[big] < 0.0:
            fied = [-z for z in fied]
    labels = [1 if z >= 0.0 else 0 for z in fied]
    return RichResult(
        title="Fiedler two-way spectral partition",
        summary_lines=[("algebraic connectivity", vals[1] if n > 1 else float("nan"))],
        payload={
            "labels": labels,
            "estimate": vals[1] if n > 1 else float("nan"),
            "fiedler": fied,
            "eigenvalues": vals,
            "n": n,
            "method": "Two-way spectral partition by the sign of the Fiedler vector",
        },
    )


def cheatsheet():
    return "sgtsbm: Two-cluster spectral partition via Fiedler sign"
