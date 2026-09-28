# morie.fn -- function file (rootcoder007/morie)
"""Network measures on edge lists: centralities, distances, cohesion, modularity, connectivity and vulnerability."""

from __future__ import annotations

import heapq
import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult

__all__ = [
    "shortest_path_lengths",
    "centralities",
    "network_summary",
    "modularity_score",
    "network_connectivity",
    "node_vulnerability",
]

INF = float("inf")


def _adj(n, edges, directed):
    """Adjacency dict of neighbour -> weight (parallel edges summed), from (u, v[, w]) tuples."""
    out = [dict() for _ in range(n)]
    for e in edges:
        u, v = int(e[0]), int(e[1])
        w = float(e[2]) if len(e) > 2 else 1.0
        if u == v:
            continue
        out[u][v] = out[u].get(v, 0.0) + w
        if not directed:
            out[v][u] = out[v].get(u, 0.0) + w
    return out


def _dijkstra(adj, s, weighted):
    d = [INF] * len(adj)
    d[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > d[u]:
            continue
        for v, w in adj[u].items():
            nd = du + (w if weighted else 1.0)
            if nd < d[v]:
                d[v] = nd
                heapq.heappush(pq, (nd, v))
    return d


def shortest_path_lengths(n: int, edges, *, directed: bool = False, weighted: bool = False):
    r"""All-pairs shortest path lengths (Dijkstra 1959), ``inf`` when unreachable, as ``igraph::distances``.

    With ``weighted`` the third element of each edge is its length.

    Examples
    --------
    >>> shortest_path_lengths(3, [(0, 1), (1, 2)])[0]
    [0.0, 1.0, 2.0]
    """
    adj = _adj(n, edges, directed)
    return [_dijkstra(adj, s, weighted) for s in range(n)]


def _brandes(adj, weighted):
    """Brandes (2001) betweenness (sum over ordered pairs; halve for undirected)."""
    n = len(adj)
    cb = [0.0] * n
    for s in range(n):
        S, P, sigma = [], [[] for _ in range(n)], [0.0] * n
        sigma[s] = 1.0
        d = [INF] * n
        d[s] = 0.0
        pq = [(0.0, s, s)]
        seen = [False] * n
        while pq:
            dv, v, pred = heapq.heappop(pq)
            if seen[v]:
                continue
            seen[v] = True
            S.append(v)
            for w, wt in adj[v].items():
                nd = dv + (wt if weighted else 1.0)
                if nd < d[w] - 1e-12 * max(1.0, abs(nd)):
                    d[w] = nd
                    sigma[w] = sigma[v]
                    P[w] = [v]
                    heapq.heappush(pq, (nd, w, v))
                elif abs(nd - d[w]) <= 1e-12 * max(1.0, abs(nd)) and not seen[w]:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        delta = [0.0] * n
        for w in reversed(S):
            for v in P[w]:
                delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]
    return cb


def centralities(n: int, edges, *, directed: bool = False, weighted: bool = False, damping: float = 0.85) -> RichResult:
    r"""Node centralities with igraph's conventions.

    ``degree`` (out-degree for directed graphs; ``strength`` sums weights),
    ``closeness`` ``1 / sum_j d_ij`` over reachable ``j`` (``igraph::closeness``),
    ``betweenness`` (Brandes 2001; unordered pairs for undirected graphs),
    ``eigenvector`` (leading eigenvector of the undirected adjacency matrix
    scaled to maximum 1; Bonacich 1972), ``pagerank`` (damping ``damping``,
    dangling nodes redistributed uniformly; Brin and Page 1998), and
    ``information`` centrality (Stephenson and Zelen 1989): with ``C = (L +
    J)^{-1}`` for the Laplacian ``L`` and all-ones ``J``, ``I_i = 1 / (C_ii +
    (tr C - 2 sum_j C_ij) / n)`` (undirected, connected).

    References
    ----------
    Brandes, U. (2001). A faster algorithm for betweenness centrality.
    *Journal of Mathematical Sociology*, 25(2), 163-177.
    Brin, S. and Page, L. (1998). The anatomy of a large-scale hypertextual
    web search engine. *Computer Networks and ISDN Systems*, 30(1-7),
    107-117.
    Stephenson, K. and Zelen, M. (1989). Rethinking centrality: methods and
    examples. *Social Networks*, 11(1), 1-37.

    Examples
    --------
    >>> c = centralities(4, [(0, 1), (1, 2), (2, 3)])
    >>> c.degree, c.betweenness
    ([1, 2, 2, 1], [0.0, 2.0, 2.0, 0.0])
    """
    adj = _adj(n, edges, directed)
    deg = [len(a) for a in adj]
    strength = [ssum(a.values()) for a in adj]
    D = [_dijkstra(adj, s, weighted) for s in range(n)]
    clo = []
    for i in range(n):
        tot = ssum(d for j, d in enumerate(D[i]) if j != i and d < INF)
        clo.append(1.0 / tot if tot > 0 else float("nan"))
    bt = _brandes(adj, weighted)
    if not directed:
        bt = [v / 2.0 for v in bt]
    # eigenvector centrality by power iteration on the (weighted) adjacency matrix
    x = [1.0] * n
    for _ in range(10000):
        y = [ssum(w * x[j] for j, w in adj[i].items()) for i in range(n)]
        m = max(y)
        if m <= 0:
            break
        y = [v / m for v in y]
        if max(abs(a - b) for a, b in zip(x, y)) < 1e-15:
            x = y
            break
        # average consecutive iterates to damp bipartite oscillation
        x = [(a + b) / 2.0 for a, b in zip(x, y)]
    # PageRank by solving (I - d P') r = (1 - d)/n + dangling mass
    outw = [ssum(a.values()) for a in adj]
    r = [1.0 / n] * n
    for _ in range(100000):
        dang = ssum(r[i] for i in range(n) if outw[i] == 0)
        nr = [(1 - damping) / n + damping * dang / n] * n
        for i in range(n):
            if outw[i] > 0:
                for j, w in adj[i].items():
                    nr[j] += damping * r[i] * w / outw[i]
        s = ssum(nr)
        nr = [v / s for v in nr]
        if max(abs(a - b) for a, b in zip(r, nr)) < 1e-16:
            r = nr
            break
        r = nr
    info = [float("nan")] * n
    if not directed:
        Lp = [[(strength[i] if i == j else -adj[i].get(j, 0.0)) + 1.0 for j in range(n)] for i in range(n)]
        try:
            C = [[float(v) for v in row] for row in inverse(Lp)]
            T = ssum(C[i][i] for i in range(n))
            info = [1.0 / (C[i][i] + (T - 2.0 * ssum(C[i])) / n) for i in range(n)]
        except (ValueError, ZeroDivisionError, ArithmeticError):
            pass
    return RichResult(
        payload={
            "degree": deg,
            "strength": strength,
            "closeness": clo,
            "betweenness": bt,
            "eigenvector": x,
            "pagerank": r,
            "information": info,
        }
    )


def network_summary(n: int, edges, *, directed: bool = False) -> RichResult:
    r"""Whole-network measures with igraph's conventions (unweighted).

    ``density`` ``m / (n(n-1)/2)`` (``n(n-1)`` if directed), eccentricities,
    ``diameter`` (largest finite distance) and ``radius`` (smallest
    eccentricity), global ``transitivity`` ``3 triangles / connected
    triples`` and local clustering (``nan`` below degree 2) of the undirected
    graph, ``reciprocity`` (share of directed edges reciprocated),
    degree ``assortativity`` (Newman 2002), global ``efficiency`` (mean ``1/d``
    over ordered pairs; Latora and Marchiori 2001) and the Shannon
    ``degree_entropy`` of the degree distribution.

    References
    ----------
    Newman, M. E. J. (2002). Assortative mixing in networks. *Physical
    Review Letters*, 89(20), 208701.
    Latora, V. and Marchiori, M. (2001). Efficient behavior of small-world
    networks. *Physical Review Letters*, 87(19), 198701.

    Examples
    --------
    >>> s = network_summary(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    >>> s.diameter, round(s.transitivity, 6), s.density
    (2.0, 0.6, 0.6666666666666666)
    """
    adj = _adj(n, edges, directed)
    und = _adj(n, edges, False)
    m = sum(len(a) for a in adj) if directed else sum(len(a) for a in adj) // 2
    dens = m / (n * (n - 1)) if directed else m / (n * (n - 1) / 2)
    D = [_dijkstra(adj, s, False) for s in range(n)]
    ecc = [max((d for d in row if d < INF), default=0.0) for row in D]
    tri = triples = 0
    local = []
    for i in range(n):
        nb = sorted(und[i])
        k = len(nb)
        t = sum(1 for a in range(k) for b in range(a + 1, k) if nb[b] in und[nb[a]])
        tri += t
        triples += k * (k - 1) / 2
        local.append(t / (k * (k - 1) / 2) if k >= 2 else float("nan"))
    rec = float("nan")
    if directed:
        e = [(u, v) for u in range(n) for v in adj[u]]
        rec = sum(1 for u, v in e if u in adj[v]) / len(e) if e else float("nan")
    # assortativity over both ends of every undirected edge (Newman 2002, eq. 4)
    deg = [len(a) for a in (und if not directed else adj)]
    if directed:
        indeg = [sum(1 for w in range(n) if v in adj[w]) for v in range(n)]
        pairs = [(len(adj[u]), indeg[v]) for u in range(n) for v in adj[u]]
    else:
        pairs = [(deg[u], deg[v]) for u in range(n) for v in und[u]]
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = ssum(xs) / len(xs), ssum(ys) / len(ys)
    sxy = ssum((a - mx) * (b - my) for a, b in pairs)
    sxx = ssum((a - mx) ** 2 for a in xs)
    syy = ssum((b - my) ** 2 for b in ys)
    assort = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    eff = ssum(1.0 / D[i][j] for i in range(n) for j in range(n) if i != j and D[i][j] < INF) / (n * (n - 1))
    counts = {}
    for d in (len(a) for a in und):
        counts[d] = counts.get(d, 0) + 1
    ent = -ssum(c / n * math.log(c / n) for c in counts.values())
    return RichResult(
        payload={
            "density": dens,
            "eccentricity": ecc,
            "diameter": max(ecc),
            "radius": min(ecc),
            "transitivity": tri / triples if triples else float("nan"),
            "local_transitivity": local,
            "reciprocity": rec,
            "assortativity": assort,
            "efficiency": eff,
            "degree_entropy": ent,
        }
    )


def modularity_score(n: int, edges, membership, *, resolution: float = 1.0) -> float:
    r"""Newman-Girvan modularity ``Q = (1/2m) sum_ij (A_ij - gamma k_i k_j / 2m) delta(c_i, c_j)`` (undirected), as ``igraph::modularity``.

    References
    ----------
    Newman, M. E. J. and Girvan, M. (2004). Finding and evaluating community
    structure in networks. *Physical Review E*, 69(2), 026113.

    Examples
    --------
    >>> round(modularity_score(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)], [0, 0, 0, 1, 1, 1]), 6)
    0.357143
    """
    adj = _adj(n, edges, False)
    k = [ssum(a.values()) for a in adj]
    two_m = ssum(k)
    c = list(membership)
    q = 0.0
    for i in range(n):
        for j in range(n):
            if c[i] == c[j]:
                q += adj[i].get(j, 0.0) - resolution * k[i] * k[j] / two_m
    return q / two_m


def _maxflow(cap, s, t):
    n = len(cap)
    flow = 0
    res = [row[:] for row in cap]
    while True:
        par = [-1] * n
        par[s] = s
        q = [s]
        while q and par[t] < 0:
            u = q.pop(0)
            for v in range(n):
                if par[v] < 0 and res[u][v] > 0:
                    par[v] = u
                    q.append(v)
        if par[t] < 0:
            return flow
        b, v = INF, t
        while v != s:
            b = min(b, res[par[v]][v])
            v = par[v]
        v = t
        while v != s:
            res[par[v]][v] -= b
            res[v][par[v]] += b
            v = par[v]
        flow += b


def network_connectivity(n: int, edges) -> RichResult:
    r"""Edge and vertex connectivity of an undirected graph (Menger's theorem via unit-capacity max flows), as igraph.

    Edge connectivity is the smallest ``s``-``t`` maximum flow; vertex
    connectivity the smallest over non-adjacent pairs of the flow in the
    node-split graph (``n - 1`` for a complete graph).

    Examples
    --------
    >>> c = network_connectivity(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    >>> c.edge, c.vertex
    (2, 2)
    """
    und = _adj(n, edges, False)
    cap = [[1 if j in und[i] else 0 for j in range(n)] for i in range(n)]
    edge = min(_maxflow(cap, 0, t) for t in range(1, n)) if n > 1 else 0
    split = [[0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        split[2 * i][2 * i + 1] = 1
        for j in und[i]:
            split[2 * i + 1][2 * j] = n
    vals = [_maxflow(split, 2 * s + 1, 2 * t) for s in range(n) for t in range(n) if s != t and t not in und[s]]
    vertex = min(vals) if vals else n - 1
    return RichResult(payload={"edge": edge, "vertex": vertex})


def node_vulnerability(n: int, edges) -> RichResult:
    r"""Node vulnerability ``(E - E_{-i}) / E`` (Latora and Marchiori 2005) and Burt's (1992) redundancy.

    ``E`` is the global efficiency and ``E_{-i}`` the efficiency (over the
    same ``n(n-1)`` pairs) after deleting node ``i``'s edges.  Burt's
    effective size ``k_i - 2 t_i / k_i`` and redundancy ``2 t_i / k_i`` use the
    number ``t_i`` of ties among ``i``'s ``k_i`` neighbours (undirected,
    unweighted).

    References
    ----------
    Latora, V. and Marchiori, M. (2005). Vulnerability and protection of
    infrastructure networks. *Physical Review E*, 71(1), 015103.
    Burt, R. S. (1992). *Structural Holes*. Harvard University Press.

    Examples
    --------
    >>> [round(v, 6) for v in node_vulnerability(4, [(0, 1), (1, 2), (2, 3)]).vulnerability]
    [0.423077, 0.769231, 0.769231, 0.423077]
    """
    und = _adj(n, edges, False)

    def eff(ed):
        D = shortest_path_lengths(n, ed)
        return ssum(1.0 / D[i][j] for i in range(n) for j in range(n) if i != j and D[i][j] < INF) / (n * (n - 1))

    ed = [(int(e[0]), int(e[1])) for e in edges]
    E = eff(ed)
    vul = [(E - eff([e for e in ed if i not in e])) / E for i in range(n)]
    red, es = [], []
    for i in range(n):
        nb = sorted(und[i])
        k = len(nb)
        t = sum(1 for a in range(k) for b in range(a + 1, k) if nb[b] in und[nb[a]])
        red.append(2.0 * t / k if k else float("nan"))
        es.append(k - 2.0 * t / k if k else float("nan"))
    return RichResult(payload={"vulnerability": vul, "redundancy": red, "effective_size": es})


def cheatsheet() -> str:
    return "centralities / network_summary / modularity_score / network_connectivity -> network measures (igraph conventions)."
