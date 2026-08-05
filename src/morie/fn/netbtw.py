# morie.fn -- function file (rootcoder007/morie)
"""Freeman betweenness centrality of a node."""

from . import _tail1core as C
from .btwns import betweenness

from ._richresult import RichResult

__all__ = ["betweenness_centrality", "betweennesscentrality"]


def betweenness_centrality(A, node=0):
    """Freeman betweenness centrality of one node.

        C_B(v) = sum_{s != v != t} sigma_st(v) / sigma_st

    ``sigma_st`` counts shortest s-t paths and ``sigma_st(v)`` those
    passing through ``v``.  The all-pairs accumulation is delegated to
    ``btwns.betweenness`` (Brandes 2001), which is the single
    implementation of the recursion in this package; this module only
    selects a node and applies Freeman's normalisation.

    Brandes accumulates over ORDERED pairs.  Freeman's C_B is defined
    over unordered pairs, so for a symmetric adjacency the ordered sum
    is halved.  The normalised score divides by the maximum possible
    value ``(n-1)(n-2)/2``, attained by the hub of a star.

    The stub this replaces took a leading ``y`` data argument that its
    body only averaged; it carried no meaning here and has been dropped.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency matrix; non-zero entries are edges (unweighted).
    node : int, default 0
        Node whose centrality is returned.

    Returns
    -------
    RichResult
        ``estimate`` (C_B of ``node``), ``normalized``, ``cb_ordered``,
        ``node``, ``n``.

    References
    ----------
    Freeman, L. C. (1977), "A set of measures of centrality based on
    betweenness", Sociometry 40(1), 35-41, doi:10.2307/3033543.
    Brandes, U. (2001), "A faster algorithm for betweenness centrality",
    Journal of Mathematical Sociology 25(2), 163-177,
    doi:10.1080/0022250X.2001.9990249.
    """
    M = C.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("betweenness_centrality: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("betweenness_centrality: adjacency matrix must be square")
    v = int(node)
    if v < 0 or v >= n:
        raise ValueError("betweenness_centrality: node out of range")
    sym = all(M[i][j] == M[j][i] for i in range(n) for j in range(n))
    cb = list(betweenness(M).value)
    ordered = float(cb[v])
    est = ordered / 2.0 if sym else ordered
    denom = (n - 1) * (n - 2) / 2.0 if sym else (n - 1) * (n - 2)
    norm = est / denom if denom > 0 else 0.0
    return RichResult(payload={
        "estimate": est, "normalized": norm, "cb_ordered": ordered,
        "node": v, "n": n, "symmetric": 1.0 if sym else 0.0,
        "method": "Freeman betweenness centrality"})


betweennesscentrality = betweenness_centrality


def cheatsheet():
    return "netbtw: Freeman betweenness centrality of a node"
