# morie.fn -- function file (rootcoder007/morie)
"""Betweenness centrality by Brandes' algorithm."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['btwcent', 'sgt_betweenness_centrality']


def btwcent(A, normalise=False):
    """Betweenness centrality by Brandes' algorithm.

    The naive definition needs every shortest path enumerated; the dependency recursion needs only one BFS per source and the counts along it, which is what makes betweenness computable on anything larger than a toy. Values are for an undirected graph and each pair is therefore counted twice in the accumulation, so the total is halved at the end -- the single most common off-by-two in implementations of this measure.


    Formula: c_B(v) = sum_{s != v != t} sigma_st(v) / sigma_st, accumulated by dependency delta_s(v) = sum_w (sigma_sv/sigma_sw)(1 + delta_s(w))

    Parameters
    ----------
    A : array-like, shape (n, n)
        Undirected unweighted adjacency; non-zero means an edge.
    normalise : bool
        Divide by (n-1)(n-2)/2, the maximum possible value.

    Returns
    -------
    RichResult
        ``betweenness``, ``normalised``, ``n``.

    References
    ----------
    Brandes (2001), A faster algorithm for betweenness centrality,
    Journal of Mathematical Sociology 25:163-177.  Paywalled and not
    held locally; the dependency recursion implemented here is the
    standard published form of the algorithm.  Its output is checked in
    the batch's anchor file against brute-force enumeration of all
    shortest paths on a small graph.
    """
    A = C.mat(A)
    n = len(A)
    adj = [[j for j in range(n) if j != i and A[i][j] != 0] for i in range(n)]
    cb = [0.0] * n
    for s in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = [0.0] * n
        dist = [-1] * n
        sigma[s] = 1.0
        dist[s] = 0
        q = [s]; h = 0
        while h < len(q):
            v = q[h]; h += 1
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = [0.0] * n
        for w in reversed(stack):
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    cb = [v / 2.0 for v in cb]
    denom = (n - 1) * (n - 2) / 2.0 if n > 2 else float("nan")
    return RichResult(payload={
        "betweenness": cb,
        "normalised": [v / denom for v in cb] if denom == denom else
                      [float("nan")] * n,
        "n": n, "method": "Betweenness centrality (Brandes)"})


sgt_betweenness_centrality = btwcent


def cheatsheet():
    return "sgtbtw: Betweenness centrality by Brandes' algorithm."
