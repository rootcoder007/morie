"""Shared pieces for contiguity-constrained regionalization: neighbours, distances, SSD, tree cuts."""

import math


def neighbours(adjacency, n):
    if (
        len(adjacency) == n
        and all(len(r) == n for r in adjacency)
        and all(v in (0, 1) for r in adjacency for v in r)
        and n > 2
    ):
        return [[j for j in range(n) if adjacency[i][j] and j != i] for i in range(n)]
    return [sorted(int(j) for j in r) for r in adjacency]


def sqdist(a, b):
    s = 0.0
    for u, v in zip(a, b):
        s += (u - v) ** 2
    return s


def ssd(X, nodes):
    """Sum of squared deviations from the group mean."""
    d = len(X[0])
    tot = 0.0
    for j in range(d):
        m = 0.0
        for i in nodes:
            m += X[i][j]
        m /= len(nodes)
        for i in nodes:
            tot += (X[i][j] - m) ** 2
    return tot


def split_tree(edges, cut):
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
    return (sorted(seen), [e for e in rest if e[0] in seen]), (
        sorted(nodes - seen),
        [e for e in rest if e[0] not in seen],
    )


def partition_tree(X, n, tree, k, ok):
    """Cut a spanning tree k - 1 times, each time at the edge (over all subtrees) of largest SSD reduction
    whose two sides satisfy ``ok``; returns labels and the SSD after each cut."""
    groups = [(list(range(n)), list(tree))]
    hist = [ssd(X, list(range(n)))]
    while len(groups) < k:
        cands = []
        for gi, (nodes, edges) in enumerate(groups):
            if not edges:
                continue
            base = ssd(X, nodes)
            for ei in range(len(edges)):
                (A, _), (B, _) = split_tree(edges, ei)
                cands.append((base - ssd(X, A) - ssd(X, B), gi, ei))
        cands.sort(key=lambda t: (-t[0], t[1], t[2]))
        for gain, gi, ei in cands:
            (A, EA), (B, EB) = split_tree(groups[gi][1], ei)
            if ok(A) and ok(B):
                groups[gi] = (A, EA)
                groups.append((B, EB))
                break
        else:
            break
        tot = 0.0
        for nodes, _ in groups:
            tot += ssd(X, nodes)
        hist.append(tot)
    labels = [0] * n
    for gi, (nodes, _) in enumerate(groups):
        for i in nodes:
            labels[i] = gi
    return labels, hist


def bound_ok(n, weights, min_size, min_weight):
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    wmin = -math.inf if min_weight is None else float(min_weight)

    def ok(nodes):
        s = 0.0
        for i in nodes:
            s += w[i]
        return len(nodes) >= min_size and s >= wmin

    return ok


def canonical(labels):
    ids, out = {}, []
    for c in labels:
        out.append(ids.setdefault(c, len(ids)))
    return out


def connected_without(members, unit, nbset):
    """Whether ``members`` minus ``unit`` stays contiguous (and non-empty)."""
    rest = [i for i in members if i != unit]
    if not rest:
        return False
    rs = set(rest)
    seen, stack = {rest[0]}, [rest[0]]
    while stack:
        u = stack.pop()
        for v in nbset[u]:
            if v in rs and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(rest)


def objective_fn(X, weights, kind):
    if kind == "ssd":
        return lambda regions: sum_ssd(X, regions)
    w = [float(v) for v in weights]

    def bal(regions):
        tot = []
        for r in regions:
            s = 0.0
            for i in r:
                s += w[i]
            tot.append(s)
        m = 0.0
        for v in tot:
            m += v
        m /= len(tot)
        out = 0.0
        for v in tot:
            out += (v - m) ** 2
        return out

    return bal


def sum_ssd(X, regions):
    t = 0.0
    for r in regions:
        t += ssd(X, r)
    return t


def local_search(labels, nbset, objective, rnd, allowed, max_iter=1000):
    """AZP local search: move border units between contiguous regions while the objective falls.

    Regions are visited in a Philox-shuffled order each sweep; for each, the units of other
    regions that touch it are tried in a shuffled order and moved in when the move keeps the
    donor contiguous, passes ``allowed`` and strictly lowers the objective."""
    labels = list(labels)
    k = max(labels) + 1
    regions = [[i for i in range(len(labels)) if labels[i] == r] for r in range(k)]
    cur = objective(regions)
    for sweep in range(max_iter):
        improved = False
        for r in sorted(range(k), key=lambda q: (rnd.u(), q)):
            border = sorted({v for i in regions[r] for v in nbset[i] if labels[v] != r})
            for v in sorted(border, key=lambda q: (rnd.u(), q)):
                d = labels[v]
                if d == r or not connected_without(regions[d], v, nbset) or not any(labels[u] == r for u in nbset[v]):
                    continue
                trial = [list(g) for g in regions]
                trial[d] = [i for i in trial[d] if i != v]
                trial[r] = sorted(trial[r] + [v])
                if not allowed(trial[d]):
                    continue
                val = objective(trial)
                if val < cur - 1e-12 * max(1.0, abs(cur)):
                    regions, cur = trial, val
                    labels[v] = r
                    improved = True
        if not improved:
            return labels, cur, sweep + 1
    return labels, cur, max_iter
