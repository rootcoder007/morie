# morie.fn -- function file (rootcoder007/morie)
"""Edge-weighted graph Laplacian."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['wlaplac', 'sgt_weighted_laplacian']


def wlaplac(W):
    """Edge-weighted graph Laplacian.

    Formula: L = T - W,  T = diag(t_v) with t_v = sum_u w_uv

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric matrix of non-negative edge weights with a zero diagonal.

    Returns
    -------
    RichResult
        ``L``, ``strength``, ``T``, ``volume``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  This particular matrix is not printed in the retrieved survey and is implemented in the standard published form; it should be re-checked against Chung (1997) Chapter 1 if the book is ever added to the library.  The weighted Laplacian is the combinatorial Laplacian with the degree replaced by the vertex strength t_v, the sum of the weights of the edges at v; it reduces to D - A when every weight is 0 or 1.  Kept separate from morie.fn.sgtlap so the unweighted case cannot silently accept a weighted argument.
    """
    A = C.mat(W)
    n = len(A)
    if n == 0 or len(A[0]) != n:
        raise ValueError("W must be a non-empty square matrix")
    for i in range(n):
        for j in range(n):
            if A[i][j] < 0.0:
                raise ValueError("edge weights must be non-negative")
            if abs(A[i][j] - A[j][i]) > 0.0:
                raise ValueError("W must be symmetric")
    t = [sum(A[i]) for i in range(n)]
    L = [[(t[i] if i == j else 0.0) - A[i][j] for j in range(n)] for i in range(n)]
    return RichResult(payload={
        "L": L, "strength": t, "T": [[t[i] if i == j else 0.0 for j in range(n)]
                                     for i in range(n)],
        "volume": sum(t), "n": n, "method": "Weighted Laplacian T - W"})


sgt_weighted_laplacian = wlaplac


def cheatsheet():
    return 'sgtwlap: Edge-weighted graph Laplacian.'
