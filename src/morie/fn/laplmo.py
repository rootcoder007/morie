# morie.fn -- function file (rootcoder007/morie)
"""Eigendecomposition of a graph Laplacian, with the Fiedler vector."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['glapeig', 'laplacian_eigen', 'laplacianeigen']


def glapeig(Adj, normalized=True):
    """Eigendecomposition of a graph Laplacian, with the Fiedler vector.

    Formula: L z = lambda z, eigenvalues in increasing order; the Fiedler vector is the eigenvector of the second smallest eigenvalue

    Parameters
    ----------
    Adj : array-like, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
    normalized : bool
        Decompose the normalized Laplacian I - D^-1/2 A D^-1/2 rather than the combinatorial D - A.

    Returns
    -------
    RichResult
        ``eigenvalues``, ``eigenvectors``, ``fiedler_value``, ``fiedler_vector``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2 for the normalized Laplacian; This particular matrix is not printed in the retrieved survey and is implemented in the standard published form; it should be re-checked against Chung (1997) Chapter 1 if the book is ever added to the library.  Eigenvalues are returned in increasing order and each eigenvector is sign-fixed so its largest-magnitude entry is positive, which is what makes the two language arms agree.  A repeated eigenvalue leaves its eigenvectors determined only up to a rotation within the eigenspace, so on a graph with a repeated Fiedler value the vector is not a stable quantity in either language.
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

    if normalized:
        s = [0.0 if d[i] == 0.0 else 1.0 / math.sqrt(d[i]) for i in range(n)]
        L = [[(1.0 if i == j else 0.0) - s[i] * A[i][j] * s[j] for j in range(n)]
             for i in range(n)]
    else:
        L = [[(d[i] if i == j else 0.0) - A[i][j] for j in range(n)]
             for i in range(n)]
    vals, vecs = C.eigsym(L)
    order = list(range(n - 1, -1, -1))
    lam = [vals[i] for i in order]
    V = [[vecs[r][i] for i in order] for r in range(n)]
    if n < 2:
        raise ValueError("the Fiedler vector needs at least two vertices")
    return RichResult(payload={
        "eigenvalues": lam, "eigenvectors": V, "fiedler_value": lam[1],
        "fiedler_vector": [V[r][1] for r in range(n)], "n": n,
        "method": "Laplacian eigendecomposition with Fiedler vector"})


laplacian_eigen = glapeig
laplacianeigen = glapeig


def cheatsheet():
    return 'laplmo: Eigendecomposition of a graph Laplacian, with the Fiedler vector.'
