# morie.fn -- function file (rootcoder007/morie)
"""Newman-Girvan modularity Q (alias of :mod:`sgtmodq`)."""

from .sgtmodq import sgt_modularity_q

__all__ = ["modularity_q", "modularityq"]


def modularity_q(G, communities):
    """Newman-Girvan modularity ``Q`` of a partition of a graph.

    This module is an ALIAS.  Modularity is implemented once, in
    ``sgtmodq.sgt_modularity_q``; this entry point delegates.  No second
    copy of the arithmetic exists.

        Q = (1 / 2m) sum_ij (A_ij - k_i k_j / 2m) delta(c_i, c_j)

    with ``2m = sum_ij A_ij`` and ``k_i = sum_j A_ij``.  The null term
    ``k_i k_j / 2m`` is the configuration-model expectation, which is
    why a single all-node community scores exactly zero however dense
    the graph.

    Parameters
    ----------
    G : array-like, shape (n, n)
        Symmetric adjacency or weight matrix.
    communities : array-like of int, length n
        Community label per node.

    Returns
    -------
    RichResult
        ``Q``, ``estimate``, ``n_communities``, ``n``.

    References
    ----------
    Newman, M. E. J. and Girvan, M. (2004), "Finding and evaluating
    community structure in networks", Physical Review E 69, 026113,
    doi:10.1103/PhysRevE.69.026113.
    """
    return sgt_modularity_q(G, communities)


modularityq = modularity_q


def cheatsheet():
    return "modulq: Newman-Girvan modularity Q (alias of sgtmodq)"
