# morie.fn -- function file (rootcoder007/morie)
"""Weighted graph Laplacian built from an edge list."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["wgtlap", "sgt_weighted_laplacian"]


def wgtlap(edges, n=None):
    """Assemble the weighted Laplacian from an edge list.

    Same operator as ``sgtlap``, reached from the representation
    people actually hold data in.  Parallel edges accumulate (their
    weights add), which is the behaviour Chung's w(u, v) already
    implies, and a loop contributes to the degree but not to the
    diagonal of L -- the d_v - w(v, v) term.

    ``edges`` rows are (u, v, weight) with ONE-BASED vertex labels,
    matching the R arm.

    Formula: w(u, v) = sum of the weights on {u, v};
             L(u, v) = d_v - w(v, v) if u = v, -w(u, v) otherwise

    Parameters
    ----------
    edges : array-like, shape (m, 3)
        Rows (u, v, weight); u, v one-based; weight non-negative.
    n : int, optional
        Number of vertices (default: the largest label seen).

    Returns
    -------
    RichResult
        ``W``, ``L``, ``degree``, ``volume``, ``n``, ``m``.

    References
    ----------
    Chung (1997), Spectral Graph Theory, CBMS 92, Section 1.4, for the
    weighted definition with loops.  Fetched from the author's own copy
    of the chapter.
    """
    E = C.mat(edges)
    if E and len(E[0]) != 3:
        raise ValueError("edges rows must be (u, v, weight)")
    m = len(E)
    hi = 0
    for r in E:
        hi = max(hi, int(r[0]), int(r[1]))
    N = hi if n is None else int(n)
    if N < 1:
        raise ValueError("the graph needs at least one vertex")
    W = [[0.0] * N for _ in range(N)]
    for r in E:
        u = int(r[0]) - 1
        v = int(r[1]) - 1
        w = float(r[2])
        if w < 0:
            raise ValueError("weights must be non-negative")
        if not (0 <= u < N and 0 <= v < N):
            raise ValueError("vertex label out of range 1..n")
        W[u][v] += w
        if u != v:
            W[v][u] += w
    d = [sum(W[i]) for i in range(N)]
    L = [[(d[i] - W[i][i]) if i == j else -W[i][j] for j in range(N)]
         for i in range(N)]
    return RichResult(payload={
        "W": W, "L": L, "degree": d, "volume": sum(d), "n": N, "m": m,
        "method": "Weighted Laplacian from an edge list"})


sgt_weighted_laplacian = wgtlap


def cheatsheet():
    return "sgtwlap: edge list -> W -> L = T - W (loops keep d_v - w(v,v))"
