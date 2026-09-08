# morie.fn -- function file (rootcoder007/morie)
"""Chung normalised Laplacian."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["normlap", "sgt_normalised_laplacian"]


def normlap(W):
    """Normalised Laplacian in Chung's sense.

    The normalisation is what makes the eigenvalues comparable across
    graphs of different degree: they always lie in [0, 2], with 0 of
    multiplicity equal to the number of connected components.  Chung's
    convention T^-1(v, v) = 0 for an isolated vertex is followed
    exactly, so an isolated vertex contributes a zero row and column
    rather than a division by zero.

    Formula: Lcal = T^-1/2 L T^-1/2, i.e.
             Lcal(u, v) = 1 - w(v, v)/d_v  if u = v and d_v != 0,
                        = -w(u, v)/sqrt(d_u d_v)  if u ~ v,
                        = 0  otherwise

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative weight matrix.

    Returns
    -------
    RichResult
        ``Lcal``, ``degree``, ``isolated`` (one-based), ``n``.

    References
    ----------
    Chung (1997), Spectral Graph Theory, CBMS 92, Sections 1.2 and 1.4:
    "L = T^-1/2 L T^-1/2 with the convention T^-1(v, v) = 0 for
    d_v = 0".  Fetched from the author's own copy of the chapter.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    d = [sum(W[i]) for i in range(n)]
    s = [0.0 if d[i] == 0.0 else 1.0 / math.sqrt(d[i]) for i in range(n)]
    L = [[(d[i] - W[i][i]) if i == j else -W[i][j] for j in range(n)]
         for i in range(n)]
    Lc = [[s[i] * L[i][j] * s[j] for j in range(n)] for i in range(n)]
    return RichResult(payload={
        "Lcal": Lc, "degree": d,
        "isolated": [i + 1 for i in range(n) if d[i] == 0.0], "n": n,
        "method": "Normalised Laplacian T^-1/2 L T^-1/2"})


sgt_normalised_laplacian = normlap


def cheatsheet():
    return "sgtnlap: Lcal = T^-1/2 (T-A) T^-1/2, zero row for isolated v"
