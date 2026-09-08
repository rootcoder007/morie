# morie.fn -- function file (rootcoder007/morie)
"""Degree matrix and volume of a graph."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['degmat', 'sgt_degree_matrix']


def degmat(Adj):
    """Degree matrix and volume of a graph.

    Formula: D = diag(d_1, ..., d_n),  d_v = sum_u A_uv;  vol(G) = sum_v d_v

    Parameters
    ----------
    Adj : array-like, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.

    Returns
    -------
    RichResult
        ``degrees``, ``D``, ``volume``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  D is the diagonal degree matrix; vol(S) = sum over v in S of d_v, so vol(G) is the sum of all degrees, which is twice the number of edges for an unweighted graph.
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

    D = [[d[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    return RichResult(payload={
        "degrees": d, "D": D, "volume": sum(d), "n": n,
        "method": "Degree matrix and graph volume"})


sgt_degree_matrix = degmat


def cheatsheet():
    return 'sgtdeg: Degree matrix and volume of a graph.'
