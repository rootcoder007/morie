# morie.fn -- function file (rootcoder007/morie)
"""Newman-Girvan modularity Q of a labelled partition."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["sgt_modularity_q"]


def sgt_modularity_q(A, labels):
    """Modularity of the partition ``labels`` on the weighted graph ``A``.

    Modularity contrasts the observed weight inside communities with the
    weight the configuration model would put there, ``k_i k_j / 2m``.
    The null term is what makes Q informative: a single community
    containing every node scores exactly zero, however dense the graph,
    because observed and expected weight then agree by construction.

    Formula: ``Q = (1 / 2m) sum_ij (A_ij - k_i k_j / 2m) delta(c_i, c_j)``
    with ``2m = sum_ij A_ij`` and ``k_i = sum_j A_ij``.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric adjacency or weight matrix.
    labels : array-like of int, length n
        Community label per node; any hashable-by-value integers.

    Returns
    -------
    RichResult
        ``Q``, ``estimate`` (the same number), ``n_communities``, ``n``.

    References
    ----------
    Newman, M. E. J. & Girvan, M. (2004).  Finding and evaluating
    community structure in networks.  Physical Review E 69, 026113.
    doi:10.1103/PhysRevE.69.026113.
    """
    M = C.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("sgt_modularity_q: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("sgt_modularity_q: adjacency matrix must be square")
    lab = [int(v) for v in C.vec(labels)]
    if len(lab) != n:
        raise ValueError("sgt_modularity_q: labels must have one entry per node")
    k = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += M[i][j]
        k[i] = s
    m2 = 0.0
    for i in range(n):
        m2 += k[i]
    if m2 <= 0.0:
        raise ValueError("sgt_modularity_q: graph has no edge weight")
    q = 0.0
    for i in range(n):
        for j in range(n):
            if lab[i] == lab[j]:
                q += M[i][j] - k[i] * k[j] / m2
    q /= m2
    comms = []
    for v in lab:
        if v not in comms:
            comms.append(v)
    return RichResult(payload={
        "Q": q, "estimate": q, "n_communities": len(comms), "n": n,
        "method": "Newman-Girvan modularity Q"})


def cheatsheet():
    return "sgtmodq: Newman-Girvan modularity Q"
