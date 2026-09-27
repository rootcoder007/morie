"""Shared pieces for the discrete location models: distances, exact enumeration, vertex substitution."""

import math
from itertools import combinations


def distances(demand=None, sites=None, dist=None):
    if dist is not None:
        D = [[float(v) for v in r] for r in dist]
    else:
        if demand is None:
            raise ValueError("give dist or demand coordinates")
        S = demand if sites is None else sites
        D = [[math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q))) for q in S] for p in demand]
    if not D or not D[0] or any(len(r) != len(D[0]) for r in D):
        raise ValueError("distance matrix must be non-empty and rectangular")
    return D


def n_comb(m, p):
    return math.comb(m, p)


def search(objective, m, p, max_enum):
    """Minimise objective(sorted site tuple) over p-subsets of range(m).

    Exact enumeration in lexicographic order when C(m, p) <= max_enum (first minimum kept);
    otherwise greedy addition followed by Teitz-Bart vertex substitution until no swap improves.
    """
    if not 1 <= p <= m:
        raise ValueError("need 1 <= p <= number of candidate sites")
    if n_comb(m, p) <= max_enum:
        best, bs = math.inf, None
        for S in combinations(range(m), p):
            v = objective(S)
            if v < best - 1e-12:
                best, bs = v, S
        return list(bs), best, "exact enumeration"
    S = []
    for _ in range(p):
        cand = [(objective(tuple(sorted(S + [j]))), j) for j in range(m) if j not in S]
        S.append(min(cand)[1])
    S = sorted(S)
    best = objective(tuple(S))
    improved = True
    while improved:
        improved = False
        for out in list(S):
            for j in range(m):
                if j in S:
                    continue
                T = tuple(sorted([s for s in S if s != out] + [j]))
                v = objective(T)
                if v < best - 1e-12:
                    S, best, improved = list(T), v, True
                    break
            if improved:
                break
    return S, best, "greedy + vertex substitution (Teitz and Bart 1968)"
