# morie.fn -- function file (rootcoder007/morie)
"""Barabasi-Albert preferential attachment graph."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['bamodel', 'barabasi_albert']


def bamodel(n, m=2, m0=None, seed=1):
    """Barabasi-Albert preferential attachment graph.

    Growth and preferential attachment together are what produce the power law: the paper shows that dropping either one gives an exponential degree distribution instead. The seed graph is a complete graph on m0 vertices -- the paper says only 'a small number m0 of vertices', and Pi is undefined if they all have degree zero, so the choice is made explicit here rather than left implied. Targets for each new vertex are drawn without replacement by inverse-CDF sampling from the shared minstd stream, one uniform per edge, so both language arms build the same graph.


    Formula: Pi(k_i) = k_i / sum_j k_j; each new vertex brings m edges

    Parameters
    ----------
    n : int
        Final number of vertices.
    m : int
        Edges brought by each new vertex.
    m0 : int, optional
        Size of the complete seed graph; ``m + 1`` if omitted.
    seed : int
        Seed of the shared minstd stream.

    Returns
    -------
    RichResult
        ``degree``, ``mean_degree``, ``max_degree``, ``edges``, ``n``, ``m``.

    References
    ----------
    Barabasi and Albert (1999), Emergence of scaling in random networks,
    Science 286:509-512, arXiv:cond-mat/9910332.  Verified against the
    paper for Pi(k_i) = k_i / sum_j k_j and the growth rule.
    """
    n = int(n); m = int(m)
    m0 = int(m0) if m0 is not None else m + 1
    if m < 1 or m0 < m or n < m0:
        raise ValueError("need 1 <= m <= m0 <= n")
    deg = [m0 - 1] * m0
    edges = [(i, j) for i in range(m0) for j in range(i + 1, m0)]
    g = C.Lcg(seed)
    for v in range(m0, n):
        cand = list(range(v))
        w = [float(deg[c]) for c in cand]
        targets = []
        for _ in range(m):
            tot = sum(w)
            u = g.unif() * tot
            acc, pick = 0.0, len(cand) - 1
            for i in range(len(cand)):
                acc += w[i]
                if u < acc:
                    pick = i
                    break
            targets.append(cand[pick])
            w[pick] = 0.0
        deg.append(m)
        for t in targets:
            deg[t] += 1
            edges.append((t, v))
    return RichResult(payload={
        "degree": deg, "mean_degree": sum(deg) / n,
        "max_degree": max(deg), "edges": edges, "n": n, "m": m,
        "method": "Barabasi-Albert preferential attachment"})


barabasi_albert = bamodel


def cheatsheet():
    return "barabsi: Barabasi-Albert preferential attachment graph."
