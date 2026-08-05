# morie.fn -- function file (rootcoder007/morie)
"""Freeman degree centrality of a node."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["degree_centrality", "degreecentrality"]


def degree_centrality(A, node=0):
    """Freeman degree centrality, normalised.

        C_D(v) = deg(v) / (n - 1)

    The divisor ``n - 1`` is the largest degree any node can have in a
    simple graph on ``n`` nodes, so ``C_D`` lands in ``[0, 1]`` and is
    comparable across graphs of different size; the hub of a star scores
    exactly 1 and an isolate exactly 0.  Weighted adjacency matrices are
    summed rather than counted, giving the weighted degree.

    The stub this replaces took a leading ``y`` data argument that its
    body only averaged; it carried no meaning here and has been dropped.
    ``degcen.degree_centrality`` is the same measure but is still a
    placeholder at the time of writing, so the arithmetic lives here.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency or weight matrix.
    node : int, default 0
        Node whose centrality is returned.

    Returns
    -------
    RichResult
        ``estimate`` (normalised C_D), ``degree`` (raw), ``node``, ``n``.

    References
    ----------
    Freeman, L. C. (1979), "Centrality in social networks: conceptual
    clarification", Social Networks 1(3), 215-239,
    doi:10.1016/0378-8733(78)90021-7.
    """
    M = C.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("degree_centrality: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("degree_centrality: adjacency matrix must be square")
    v = int(node)
    if v < 0 or v >= n:
        raise ValueError("degree_centrality: node out of range")
    if n == 1:
        raise ValueError("degree_centrality: undefined for a single-node graph")
    deg = 0.0
    for j in range(n):
        if j != v:
            deg += M[v][j]
    return RichResult(payload={
        "estimate": deg / (n - 1), "degree": deg, "node": v, "n": n,
        "method": "Freeman degree centrality (normalised)"})


degreecentrality = degree_centrality


def cheatsheet():
    return "netdeg: Freeman degree centrality of a node (normalised)"
