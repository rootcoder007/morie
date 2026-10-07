"""SKATER: spatial 'K'luster analysis by tree edge removal (Assuncao, Neves, Camara and Freitas 2006).

Assuncao, R. M., Neves, M. C., Camara, G. and da Costa Freitas, C. (2006). Efficient
regionalization techniques for socio-economic geographical units using minimum spanning
trees. International Journal of Geographical Information Science 20, 797-811.
"""

import math

from ._richresult import RichResult

__all__ = ["skater"]


def _ssd(X, nodes, squared):
    d = len(X[0])
    mu = []
    for j in range(d):
        s = 0.0
        for i in nodes:
            s += X[i][j]
        mu.append(s / len(nodes))
    tot = 0.0
    for i in nodes:
        q = 0.0
        for j in range(d):
            q += (X[i][j] - mu[j]) ** 2
        tot += q if squared else math.sqrt(q)
    return tot


def _split(edges, cut):
    """Remove edge ``cut`` from a tree given as an edge list; return the two node sets and edge lists."""
    a, b = edges[cut]
    rest = [e for k, e in enumerate(edges) if k != cut]
    adj = {}
    for u, v in rest:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen, stack = {a}, [a]
    while stack:
        u = stack.pop()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    nodes = set(adj) | {a, b}
    A = sorted(seen)
    B = sorted(nodes - seen)
    return (A, [e for e in rest if e[0] in seen]), (B, [e for e in rest if e[0] not in seen])


def skater(X, adjacency, k, min_size=1, weights=None, min_weight=None, squared=False, start=0):
    r"""Contiguity-constrained clustering into k regions by pruning a minimum spanning tree.

    The attribute distance d_ij = ||x_i - x_j|| weights each contiguity edge; Prim's
    algorithm from ``start`` gives the minimum spanning tree. Each of the k - 1 cuts takes,
    over every current subtree, the edge whose removal most reduces the within-group
    dissimilarity SSW(T) - SSW(T_a) - SSW(T_b), trying candidates in decreasing order until
    both parts respect ``min_size`` (and ``min_weight`` on the summed ``weights``).
    SSW is the sum of Euclidean distances to the group mean, as in spdep::skater; set
    ``squared=True`` for the sum of squared deviations of Assuncao et al. (2006).

    Parameters
    ----------
    X : list of attribute vectors (n x d)
    adjacency : n x n 0/1 matrix or list of neighbour lists
    k : int
        Number of regions.
    min_size : int
    weights : sequence, optional
        Per-unit weights for the ``min_weight`` floor (e.g. population).
    min_weight : float, optional
    squared : bool
    start : int
        Root of Prim's algorithm.

    Returns
    -------
    RichResult
        Keys: labels (0-based region per unit), mst (edges), ssw (total within-region
        dissimilarity after each cut), n_regions.

    References
    ----------
    Assuncao, R. M. et al. (2006). International Journal of Geographical Information Science 20, 797-811.

    Examples
    --------
    >>> X = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]]
    >>> path = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]]
    >>> skater(X, path, 2)["labels"]
    [0, 0, 0, 1, 1, 1]
    """
    X = [[float(v) for v in row] for row in X]
    n = len(X)
    if (
        len(adjacency) == n
        and all(len(r) == n for r in adjacency)
        and all(v in (0, 1) for r in adjacency for v in r)
        and n > 2
    ):
        nb = [[j for j in range(n) if adjacency[i][j] and j != i] for i in range(n)]
    else:
        nb = [sorted(int(j) for j in r) for r in adjacency]
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    wmin = -math.inf if min_weight is None else float(min_weight)
    # Prim (as spdep::mstree): repeatedly attach the cheapest outside node, ties to the lowest index
    inside = [False] * n
    best = [math.inf] * n
    par = [-1] * n
    inside[start] = True
    for j in nb[start]:
        best[j] = math.dist(X[start], X[j])
        par[j] = start
    mst = []
    for _ in range(n - 1):
        cand = [i for i in range(n) if not inside[i]]
        u = min(cand, key=lambda i: (best[i], i))
        if best[u] == math.inf:
            raise ValueError("the contiguity graph is not connected")
        inside[u] = True
        mst.append((par[u], u))
        for j in nb[u]:
            dj = math.dist(X[u], X[j])
            if not inside[j] and dj < best[j]:
                best[j], par[j] = dj, u
    groups = [(list(range(n)), list(mst))]

    def ok(nodes):
        s = 0.0
        for i in nodes:
            s += w[i]
        return len(nodes) >= min_size and s >= wmin

    hist = [_ssd(X, list(range(n)), squared)]
    frozen = set()
    while len(groups) < k:
        cands = []
        for gi, (nodes, edges) in enumerate(groups):
            if gi in frozen or not edges:
                continue
            base = _ssd(X, nodes, squared)
            for ei in range(len(edges)):
                (A, _), (B, _) = _split(edges, ei)
                cands.append((base - _ssd(X, A, squared) - _ssd(X, B, squared), gi, ei))
        if not cands:
            break
        cands.sort(key=lambda t: (-t[0], t[1], t[2]))
        done = False
        for _gain, gi, ei in cands:
            (A, EA), (B, EB) = _split(groups[gi][1], ei)
            if ok(A) and ok(B):
                groups[gi] = (A, EA)
                groups.append((B, EB))
                done = True
                break
        if not done:
            break
        tot = 0.0
        for nodes, _ in groups:
            tot += _ssd(X, nodes, squared)
        hist.append(tot)
    labels = [0] * n
    for gi, (nodes, _) in enumerate(groups):
        for i in nodes:
            labels[i] = gi
    return RichResult(
        title="SKATER regionalization",
        summary_lines=[("regions", len(groups))],
        payload={"labels": labels, "mst": mst, "ssw": hist, "n_regions": len(groups)},
    )


def cheatsheet():
    return "skater: SKATER regionalization by minimum-spanning-tree edge removal"
