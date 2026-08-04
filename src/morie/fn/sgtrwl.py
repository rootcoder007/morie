# morie.fn -- function file (rootcoder007/morie)
"""Random-walk transition matrix and random-walk Laplacian."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['rwlaplac', 'sgt_random_walk_laplacian']


def rwlaplac(Adj):
    """Random-walk transition matrix and random-walk Laplacian.

    Formula: W = D^-1 A;  L_rw = I - W

    Parameters
    ----------
    Adj : array-like, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.

    Returns
    -------
    RichResult
        ``W``, ``L_rw``, ``degrees``, ``stationary``, ``isolated``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2: the transition probability matrix is W = D^-1 A, which is not symmetric, and pi = (d_1/vol(G), ..., d_n/vol(G)) is the stationary distribution of the walk on a connected non-bipartite graph.  L_rw = I - W is similar to the normalized Laplacian through L = D^1/2 (I - W) D^-1/2, so the two have the same eigenvalues.  Isolated vertices get an all-zero row of W.
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
    Wm = [[0.0 if d[i] == 0.0 else A[i][j] / d[i] for j in range(n)]
          for i in range(n)]
    Lr = [[(1.0 if i == j else 0.0) - Wm[i][j] for j in range(n)] for i in range(n)]
    vol = sum(d)
    pi = [0.0] * n if vol == 0.0 else [v / vol for v in d]
    return RichResult(payload={
        "W": Wm, "L_rw": Lr, "degrees": d, "stationary": pi, "isolated": iso,
        "n": n, "method": "Random-walk matrix D^-1 A and Laplacian I - W"})


sgt_random_walk_laplacian = rwlaplac


def cheatsheet():
    return 'sgtrwl: Random-walk transition matrix and random-walk Laplacian.'
