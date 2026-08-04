# morie.fn -- function file (rootcoder007/morie)
"""Combinatorial graph Laplacian L = T - A."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["graphlap", "sgt_laplacian"]


def graphlap(W):
    """Combinatorial (unnormalised) Laplacian of a weighted graph.

    ``rowsum`` is returned because it must be zero to machine
    precision: the all-ones vector is always in the kernel, and a
    non-zero row sum means the weight matrix was not symmetric or the
    loop convention was mishandled.

    Formula: L(u, v) = d_v - w(v, v) if u = v, -w(u, v) if u ~ v, else 0

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative weight matrix.

    Returns
    -------
    RichResult
        ``L``, ``degree``, ``rowsum``, ``n``.

    References
    ----------
    Chung (1997), Spectral Graph Theory, CBMS 92, Section 1.4, which
    generalises the Section 1.2 definition to weighted graphs with
    loops exactly as written above.  Fetched from the author's own
    copy of the chapter.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    d = [sum(W[i]) for i in range(n)]
    L = [[(d[i] - W[i][i]) if i == j else -W[i][j] for j in range(n)]
         for i in range(n)]
    return RichResult(payload={
        "L": L, "degree": d, "rowsum": [sum(r) for r in L], "n": n,
        "method": "Combinatorial Laplacian L = T - A"})


sgt_laplacian = graphlap


def cheatsheet():
    return "sgtlap: L(u,v) = d_v - w(v,v) on the diagonal, -w(u,v) off it"
