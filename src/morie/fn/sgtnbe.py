# morie.fn -- function file (rootcoder007/morie)
"""Non-backtracking (Hashimoto) matrix of a graph."""

import numpy as np

from ._richresult import RichResult

__all__ = ["sgt_nonbacktracking_matrix"]


def sgt_nonbacktracking_matrix(edges, n=None):
    r"""Hashimoto non-backtracking matrix on the directed edge set.

    Each undirected edge {u, v} becomes two directed edges. For
    directed edges e = (u, v) and f = (w, x),

    .. math:: B_{ef} = 1 \iff v = w \text{ and } x \neq u,

    i.e. f continues where e ends without immediately walking back.
    Its spectrum drives spectral clustering in sparse graphs where the
    adjacency spectrum fails (Krzakala et al. 2013): powers of B count
    non-backtracking walks, so the localisation on high-degree
    vertices that pollutes adjacency eigenvectors cannot happen.

    This replaces a placeholder that returned the mean of the edge
    array.

    Parameters
    ----------
    edges : iterable of pairs
        Undirected edge list; labels may be arbitrary hashables.
    n : int, optional
        Number of nodes (labels validated against it as in
        :func:`morie.fn.sgtadj.sgt_adjacency_matrix`).

    Returns
    -------
    RichResult
        keys: ``B`` (2m x 2m ndarray), ``directed_edges`` (row order
        of B, as index pairs), ``n``, ``m``, ``method``.

    References
    ----------
    Hashimoto, K. (1989). Zeta functions of finite graphs and
    representations of p-adic groups. *Advanced Studies in Pure
    Mathematics*, 15, 211-280.
    Krzakala, F., Moore, C., Mossel, E., Neeman, J., Sly, A.,
    Zdeborova, L. & Zhang, P. (2013). Spectral redemption in
    clustering sparse networks. *PNAS*, 110(52), 20935-20940.
    """
    from .sgtadj import sgt_adjacency_matrix

    adj = sgt_adjacency_matrix(edges, n=n, directed=False)
    A = np.asarray(adj["A"], dtype=float)
    size = int(adj["n"])

    dir_edges = [(u, v) for u in range(size) for v in range(size) if A[u, v] > 0 and u != v]
    m2 = len(dir_edges)
    B = np.zeros((m2, m2))
    pos = {e: i for i, e in enumerate(dir_edges)}
    for (u, v), i in pos.items():
        for w in range(size):
            if A[v, w] > 0 and w != u and w != v:
                B[i, pos[(v, w)]] = 1.0

    return RichResult(
        payload={
            "B": B,
            "directed_edges": dir_edges,
            "n": size,
            "m": m2 // 2,
            "method": "Non-backtracking (Hashimoto) matrix",
        }
    )


def cheatsheet():
    return "sgtnbe: non-backtracking (Hashimoto) matrix"
