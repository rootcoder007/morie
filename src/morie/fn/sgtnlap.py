# morie.fn -- function file (rootcoder007/morie)
"""Normalized graph Laplacian."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['nlaplac', 'sgt_normalised_laplacian']


def nlaplac(Adj):
    """Normalized graph Laplacian.

    Formula: L = I - D^(-1/2) A D^(-1/2)

    Parameters
    ----------
    Adj : array-like, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.

    Returns
    -------
    RichResult
        ``L``, ``degrees``, ``isolated``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2 prints L = I - D^-1/2 A D^-1/2 = D^1/2 (I - W) D^-1/2.  Isolated vertices have d_v = 0 and no D^-1/2; the convention used here is Chung's, which sets the corresponding row and column of D^-1/2 A D^-1/2 to zero, leaving a 1 on the diagonal of L.  ``isolated`` reports their 1-based indices so the caller can see the convention was applied.
    """
    A = C.mat(Adj)
    n = len(A)
    if n == 0 or len(A[0]) != n:
        raise ValueError("Adj must be a non-empty square matrix")
    for i in range(n):
        for j in range(n):
            if A[i][j] < 0.0:
                raise ValueError("edge weights must be non-negative")
            if abs(A[i][j] - A[j][i]) > 0.0:
                raise ValueError("Adj must be symmetric")
    d = [sum(A[i]) for i in range(n)]

    iso = [i + 1 for i in range(n) if d[i] == 0.0]
    s = [0.0 if d[i] == 0.0 else 1.0 / math.sqrt(d[i]) for i in range(n)]
    L = [[(1.0 if i == j else 0.0) - s[i] * A[i][j] * s[j] for j in range(n)]
         for i in range(n)]
    return RichResult(payload={
        "L": L, "degrees": d, "isolated": iso, "n": n,
        "method": "Normalized Laplacian I - D^-1/2 A D^-1/2"})


sgt_normalised_laplacian = nlaplac


def cheatsheet():
    return 'sgtnlap: Normalized graph Laplacian.'
