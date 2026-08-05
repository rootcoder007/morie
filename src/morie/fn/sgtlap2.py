# morie.fn -- function file (rootcoder007/morie)
"""Laplacian eigenmaps: embed a graph in k dimensions."""

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgt_laplacian_eigenmaps"]


def _norm_laplacian(A, who):
    M = C.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError(who + ": adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError(who + ": adjacency matrix must be square")
    d = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += M[i][j]
        if s <= 0.0:
            raise ValueError(who + ": every node must have positive degree")
        d.append(s)
    L = [[(1.0 if i == j else 0.0) - M[i][j] / math.sqrt(d[i] * d[j])
          for j in range(n)] for i in range(n)]
    return M, L, d, n


def sgt_laplacian_eigenmaps(A, k=2):
    """Embed the nodes of ``A`` using the low eigenvectors of ``L_sym``.

    The trivial eigenvector ``D^{1/2} 1`` at eigenvalue 0 carries no
    geometry and is dropped; the next ``k`` are the embedding.  On a
    graph with ``c`` connected components the eigenvalue 0 has
    multiplicity ``c``, so asking for an embedding of a disconnected
    graph returns coordinates that are constant per component -- that is
    the method behaving correctly, not a defect.

    Formula: ``L_sym = I - D^-1/2 A D^-1/2``; ``Y`` is columns 2..k+1 of
    its eigenvectors, ordered by ascending eigenvalue.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric non-negative weight matrix.
    k : int, default 2
        Embedding dimension; needs ``k <= n - 1``.

    Returns
    -------
    RichResult
        ``Y`` (n rows of k coordinates), ``eigvals`` (the k used),
        ``lambda1`` (the dropped trivial eigenvalue), ``k``, ``n``.

    References
    ----------
    Belkin, M. & Niyogi, P. (2003).  Laplacian eigenmaps for
    dimensionality reduction and data representation.  Neural
    Computation 15(6):1373-1396.  doi:10.1162/089976603321780317.
    """
    _, L, _, n = _norm_laplacian(A, "sgt_laplacian_eigenmaps")
    k = int(k)
    if k < 1 or k > n - 1:
        raise ValueError("sgt_laplacian_eigenmaps: need 1 <= k <= n - 1")
    vals, vecs = core.jacobi(L)
    Y = [[vecs[i][j + 1] for j in range(k)] for i in range(n)]
    return RichResult(payload={
        "Y": Y, "eigvals": [vals[j + 1] for j in range(k)],
        "lambda1": vals[0], "k": k, "n": n,
        "method": "Laplacian eigenmaps on L_sym"})


def cheatsheet():
    return "sgtlap2: Laplacian eigenmaps embedding"
