# morie.fn -- function file (rootcoder007/morie)
"""Combinatorial graph Laplacian."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['glaplac', 'sgt_laplacian', 'sgtlaplacian']


def glaplac(Adj):
    """Combinatorial graph Laplacian.

    Formula: L = D - A

    Parameters
    ----------
    Adj : array-like, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.

    Returns
    -------
    RichResult
        ``L``, ``degrees``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  This particular matrix is not printed in the retrieved survey and is implemented in the standard published form; it should be re-checked against Chung (1997) Chapter 1 if the book is ever added to the library.  The combinatorial Laplacian is positive semi-definite with the all-ones vector in its kernel, so its smallest eigenvalue is always zero.
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

    L = [[(d[i] if i == j else 0.0) - A[i][j] for j in range(n)] for i in range(n)]
    return RichResult(payload={
        "L": L, "degrees": d, "n": n, "method": "Combinatorial Laplacian D - A"})


sgt_laplacian = glaplac
sgtlaplacian = glaplac


def cheatsheet():
    return 'sgtlap: Combinatorial graph Laplacian.'
