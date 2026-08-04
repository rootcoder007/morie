# morie.fn -- function file (rootcoder007/morie)
"""Degree matrix and volume of a (weighted) graph."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["degmat", "sgt_degree_matrix"]


def degmat(W):
    """Degrees, degree matrix T and volume of a weighted graph.

    Everything in spectral graph theory is normalised by these
    numbers, so getting the weighted definition right once here saves
    getting it wrong in five other modules.  Isolated vertices are
    reported rather than silently dropped, because they are exactly
    the vertices the normalised Laplacian has to special-case.

    Formula: d_v = sum_u w(u, v);  T = diag(d_v);  vol G = sum_v d_v

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative weight matrix; the diagonal is the loop
        weight and does count towards the degree.

    Returns
    -------
    RichResult
        ``degree``, ``T``, ``volume``, ``isolated`` (one-based),
        ``n``.

    References
    ----------
    Chung (1997), Spectral Graph Theory, CBMS 92, Section 1.4:
    "the degree d_v of a vertex v is defined to be d_v = sum_u w(u, v)"
    and "vol G = sum_v d_v".  Fetched from the author's own copy of the
    chapter.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    for i in range(n):
        for j in range(n):
            if W[i][j] < 0:
                raise ValueError("weights must be non-negative")
            if abs(W[i][j] - W[j][i]) > 1e-12:
                raise ValueError("W must be symmetric")
    d = [sum(W[i]) for i in range(n)]
    T = [[d[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    return RichResult(payload={
        "degree": d, "T": T, "volume": sum(d),
        "isolated": [i + 1 for i in range(n) if d[i] == 0.0], "n": n,
        "method": "Degree matrix and volume"})


sgt_degree_matrix = degmat


def cheatsheet():
    return "sgtdeg: d_v = sum_u w(u,v); vol G = sum_v d_v"
