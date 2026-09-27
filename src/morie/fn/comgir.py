"""Girvan-Newman divisive community detection by edge betweenness.

Girvan, M. and Newman, M. E. J. (2002). Community structure in social and
biological networks. PNAS 99, 7821-7826. Newman, M. E. J. and Girvan, M.
(2004). Finding and evaluating community structure in networks. Physical
Review E 69, 026113 (modularity as the stopping rule).
"""

from collections import deque

from ._richresult import RichResult

__all__ = ["girvan_newman"]


def _edge_betweenness(adj, n):
    # Brandes (2001) accumulation for unweighted undirected graphs.
    eb = {}
    for s in range(n):
        stack, pred = [], [[] for _ in range(n)]
        sigma, dist = [0.0] * n, [-1] * n
        sigma[s], dist[s] = 1.0, 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                c = sigma[v] / sigma[w] * (1.0 + delta[w])
                e = (v, w) if v < w else (w, v)
                eb[e] = eb.get(e, 0.0) + c
                delta[v] += c
    return {e: b / 2.0 for e, b in eb.items()}


def _components(adj, n):
    lab, c = [-1] * n, 0
    for s in range(n):
        if lab[s] < 0:
            lab[s] = c
            stack = [s]
            while stack:
                v = stack.pop()
                for w in adj[v]:
                    if lab[w] < 0:
                        lab[w] = c
                        stack.append(w)
            c += 1
    return lab


def _modularity(edges, deg, m, lab):
    if m == 0:
        return 0.0
    inside = sum(1 for i, j in edges if lab[i] == lab[j])
    K = {}
    for v, d in enumerate(deg):
        K[lab[v]] = K.get(lab[v], 0.0) + d
    return inside / m - sum(k * k for k in K.values()) / (4.0 * m * m)


def girvan_newman(G, n_communities=None):
    """Remove the highest-betweenness edge repeatedly; keep the best split.

    Edge betweenness is recomputed (Brandes 2001) after every removal; ties are
    broken by the lexicographically smallest edge (i, j), i < j. Each time the
    number of connected components grows, the partition is scored by the
    modularity of the ORIGINAL graph, Q = sum_c [ e_c / m - (K_c / 2m)^2 ].
    The returned partition is the one of maximum Q (Newman and Girvan 2004),
    or the first with ``n_communities`` components when that is given.
    Any non-zero entry of the (symmetric) matrix is an edge.

    Parameters
    ----------
    G : square matrix
        Adjacency matrix of an undirected graph.
    n_communities : int, optional

    Returns
    -------
    RichResult
        Keys: labels, estimate (modularity of the returned partition),
        modularity, n_communities, removed (edges in removal order),
        path (list of (n_components, modularity) at each split).

    References
    ----------
    Girvan, M. and Newman, M. E. J. (2002). PNAS 99, 7821-7826.
    Newman, M. E. J. and Girvan, M. (2004). Physical Review E 69, 026113.
    Brandes, U. (2001). Journal of Mathematical Sociology 25, 163-177.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
    ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> girvan_newman(A)["labels"]
    [0, 0, 0, 1, 1, 1]
    """
    W = [[float(v) for v in row] for row in G]
    n = len(W)
    if n == 0 or any(len(row) != n for row in W):
        raise ValueError("G must be a non-empty square matrix")
    if any((W[i][j] != 0) != (W[j][i] != 0) for i in range(n) for j in range(n)):
        raise ValueError("G must be symmetric")
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if W[i][j] != 0]
    m = len(edges)
    deg = [float(sum(1 for j in range(n) if j != i and W[i][j] != 0)) for i in range(n)]
    adj = [[j for j in range(n) if j != i and W[i][j] != 0] for i in range(n)]
    lab = _components(adj, n)
    best_lab, best_q = lab, _modularity(edges, deg, m, lab)
    path = [(max(lab) + 1, best_q)]
    removed = []
    target = None if n_communities is None else int(n_communities)
    ncomp = max(lab) + 1
    while (target is None or ncomp < target) and any(adj):
        eb = _edge_betweenness(adj, n)
        top = max(eb.values())
        i, j = min(e for e, b in eb.items() if b >= top - 1e-9 * max(1.0, top))
        adj[i].remove(j)
        adj[j].remove(i)
        removed.append((i, j))
        new = _components(adj, n)
        if max(new) + 1 > ncomp:
            ncomp = max(new) + 1
            q = _modularity(edges, deg, m, new)
            path.append((ncomp, q))
            if target is not None or q > best_q + 1e-12:
                best_lab, best_q = new, q
    if target is not None and ncomp < target:
        raise ValueError("the graph cannot be split into that many communities")
    return RichResult(
        title="Girvan-Newman communities",
        summary_lines=[("communities", max(best_lab) + 1), ("modularity", best_q)],
        payload={
            "labels": best_lab,
            "estimate": best_q,
            "modularity": best_q,
            "n_communities": max(best_lab) + 1,
            "removed": removed,
            "path": path,
        },
    )


def cheatsheet():
    return "comgir: Girvan-Newman edge-betweenness divisive communities, best-modularity cut"


# compact alias per ledger/NAMING.md
girvannewman = girvan_newman
