# morie.fn -- function file (rootcoder007/morie)
"""Adjacency matrix from an edge list."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgt_adjacency_matrix"]


def sgt_adjacency_matrix(edges, n=None, directed=False):
    r"""Build the adjacency matrix of a graph from its edge list.

    .. math:: A_{ij} = 1 \iff (i, j) \in E

    with :math:`A_{ji} = 1` added for undirected graphs. Node labels
    may be arbitrary hashables; they are mapped to indices in sorted
    order and the mapping is returned. Self-loops are kept on the
    diagonal; duplicate edges collapse to a single 1.

    This replaces a placeholder that returned the mean of the edge
    array cast to float -- meaningless for a graph.

    Parameters
    ----------
    edges : iterable of pairs
        The edge list.
    n : int, optional
        Number of nodes. Inferred from the labels when omitted; must
        be at least the number of distinct labels when given (integer
        labels are then used as indices directly).
    directed : bool, default False
        Keep edges one-way.

    Returns
    -------
    RichResult
        keys: ``A`` (n x n ndarray), ``nodes`` (label -> index),
        ``degree`` (out-degree per node), ``n``, ``m``, ``directed``,
        ``method``.

    References
    ----------
    Chung, F. R. K. (1997). *Spectral Graph Theory*. CBMS Regional
    Conference Series in Mathematics 92, AMS. Ch. 1.
    """
    edges = [tuple(e) for e in edges]
    for e in edges:
        if len(e) != 2:
            raise ValueError(f"every edge must be a pair, got {e!r}.")

    labels = sorted({v for e in edges for v in e}, key=str)
    if n is None:
        index = {v: i for i, v in enumerate(labels)}
        size = len(labels)
    else:
        size = int(n)
        if all(isinstance(v, (int, np.integer)) for v in labels):
            if labels and (min(labels) < 0 or max(labels) >= size):
                raise ValueError(f"integer labels must lie in [0, {size - 1}].")
            index = {v: int(v) for v in labels}
        else:
            if len(labels) > size:
                raise ValueError(f"n={size} but {len(labels)} distinct labels.")
            index = {v: i for i, v in enumerate(labels)}

    A = np.zeros((size, size))
    for u, v in edges:
        A[index[u], index[v]] = 1.0
        if not directed:
            A[index[v], index[u]] = 1.0

    return RichResult(
        payload={
            "A": A,
            "nodes": index,
            "degree": A.sum(axis=1),
            "n": int(size),
            "m": int(len(set(map(frozenset if not directed else tuple, edges)))),
            "directed": bool(directed),
            "method": "Adjacency matrix from edge list",
        }
    )


def cheatsheet():
    return "sgtadj: adjacency matrix from an edge list"
