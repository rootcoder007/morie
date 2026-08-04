# morie.fn -- function file (rootcoder007/morie)
"""Vertex strengths of a weighted graph."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['vstrength', 'sgt_vertex_strengths']


def vstrength(W):
    """Vertex strengths of a weighted graph.

    Degree counts a vertex's neighbours; strength counts what flows through it. The two come apart exactly when weights are heterogeneous, which is the regime the paper is about, so the ratio s_i/k_i is returned alongside.


    Formula: s_i = sum_j a_ij w_ij

    Parameters
    ----------
    W : array-like, shape (n, n)
        Weighted adjacency matrix; a zero entry means no edge.

    Returns
    -------
    RichResult
        ``strength``, ``degree``, ``ratio``, ``total``, ``n``.

    References
    ----------
    Barrat, Barthelemy, Pastor-Satorras and Vespignani (2004), The
    architecture of complex weighted networks, PNAS 101:3747-3752,
    arXiv:cond-mat/0311416, equation (2).  Verified against the paper.
    """
    W = C.mat(W)
    n = len(W)
    s = [sum(W[i][j] for j in range(n) if j != i and W[i][j] != 0) for i in range(n)]
    k = [sum(1 for j in range(n) if j != i and W[i][j] != 0) for i in range(n)]
    ratio = [s[i] / k[i] if k[i] else float("nan") for i in range(n)]
    return RichResult(payload={
        "strength": s, "degree": k, "ratio": ratio, "total": sum(s), "n": n,
        "method": "Weighted-graph vertex strengths"})


sgt_vertex_strengths = vstrength


def cheatsheet():
    return "sgtvst: Vertex strengths of a weighted graph."
