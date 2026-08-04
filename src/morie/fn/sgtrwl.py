# morie.fn -- function file (rootcoder007/morie)
"""Random-walk Laplacian."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["rwlap", "sgt_random_walk_laplacian"]


def rwlap(W):
    """Random-walk Laplacian L_rw = I - P, with P the transition matrix.

    L_rw is not symmetric, so it is not the one to eigendecompose
    directly; it is here because its spectrum is identical to that of
    the symmetric normalised Laplacian (they are similar via T^1/2),
    and because P is the object every mixing-time argument is about.

    Formula: P = T^-1 W;  L_rw = I - T^-1 W

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative weight matrix.  Every vertex must have
        positive degree.

    Returns
    -------
    RichResult
        ``Lrw``, ``P``, ``degree``, ``rowsum_P``, ``n``.

    References
    ----------
    von Luxburg (2007), A Tutorial on Spectral Clustering, Statistics
    and Computing 17(4), 395-416, Section 3, which defines
    L_rw = I - D^-1 W and notes it is "closely related to a random
    walk".  Fetched from arXiv:0711.0189.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    d = [sum(W[i]) for i in range(n)]
    if any(v <= 0 for v in d):
        raise ValueError("L_rw needs every vertex to have positive degree")
    P = [[W[i][j] / d[i] for j in range(n)] for i in range(n)]
    Lrw = [[(1.0 if i == j else 0.0) - P[i][j] for j in range(n)]
           for i in range(n)]
    return RichResult(payload={
        "Lrw": Lrw, "P": P, "degree": d,
        "rowsum_P": [sum(r) for r in P], "n": n,
        "method": "Random-walk Laplacian I - T^-1 W"})


sgt_random_walk_laplacian = rwlap


def cheatsheet():
    return "sgtrwl: L_rw = I - T^-1 W"
