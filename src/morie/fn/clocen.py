# morie.fn -- function file (rootcoder007/morie)
"""Closeness centrality."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['clocent', 'closeness_centrality']


def clocent(A):
    """Closeness centrality.

    Sabidussi's sum distance inverted and scaled. Disconnected graphs make the textbook (n-1)/sum d form infinite, so the sum runs over the reachable set only and ``reachable`` is returned so the caller can see how much of the graph each score actually covers -- a score computed over three nodes is not comparable with one computed over three hundred.


    Formula: C_C(v) = (r - 1) / sum_{u reachable} d(v, u)

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency matrix; non-zero means an edge.

    Returns
    -------
    RichResult
        ``closeness``, ``reachable``, ``total_distance``, ``n``.

    References
    ----------
    Sabidussi (1966) for the sum-distance form and Freeman (1979),
    Centrality in social networks: conceptual clarification, Social
    Networks 1:215-239, for the (n-1)-normalised measure.  Freeman's
    article is paywalled; the normalisation C(v) = (n-1)/sum_u d(v,u)
    is as restated in the centrality literature that cites him.
    """
    A = C.mat(A)
    n = len(A)
    adj = [[j for j in range(n) if j != i and A[i][j] != 0] for i in range(n)]
    clos, reach, tot = [], [], []
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = [s]; h = 0
        while h < len(q):
            v = q[h]; h += 1
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
        d = [dist[t] for t in range(n) if t != s and dist[t] > 0]
        r = len(d) + 1
        tot.append(float(sum(d)))
        reach.append(r)
        clos.append((r - 1.0) / sum(d) if d else float("nan"))
    return RichResult(payload={
        "closeness": clos, "reachable": reach, "total_distance": tot, "n": n,
        "method": "Closeness centrality"})


closeness_centrality = clocent


def cheatsheet():
    return "clocen: Closeness centrality."
