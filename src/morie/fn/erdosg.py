# morie.fn -- function file (rootcoder007/morie)
"""Erdos-Renyi G(n,p)."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["erdos_renyi_gnp"]

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _u(k):
    """Deterministic uniform in [0,1) for edge slot k."""
    return core.vdc(k // len(_PRIMES) + 1, _PRIMES[k % len(_PRIMES)])


def erdos_renyi_gnp(n, p):
    """
    Erdos-Renyi G(n,p)

    Formula: each of the C(n,2) possible edges is present independently
    with probability p.  Realised here on a DETERMINISTIC low-discrepancy
    stream -- edge slot k is compared against p using van der Corput base
    ``_PRIMES[k mod 12]``, so successive dyads are not correlated the way
    one shared stream would make them -- and both language arms therefore
    build the identical graph.

    PROVENANCE: the binomial model G(n,p) is Gilbert (1959).  Erdos &
    Renyi (1959) defined G(n,M), the uniform model on graphs with exactly
    M edges; the two are asymptotically equivalent but they are not the
    same construction, and the name attached to G(n,p) in the literature
    is a misattribution.

    Alongside the realised graph the exact analytic quantities are
    returned: E[edges] = C(n,2) p, E[degree] = (n-1) p, the connectivity
    threshold log(n)/n and the giant-component threshold 1/n.

    Parameters
    ----------
    n : int
        Number of vertices (>= 1).
    p : float
        Edge probability in [0, 1].

    Returns
    -------
    result : dict
        Keys: estimate (realised density), edges, density, expected_edges,
        mean_degree, expected_degree, n_components, largest_component,
        giant_threshold, connectivity_threshold, n, method.

    References
    ----------
    Gilbert (1959), Ann. Math. Statist. 30(4):1141-1144,
    doi:10.1214/aoms/1177706098.
    Erdos & Renyi (1959), Publ. Math. Debrecen 6:290-297 (the G(n,M) model).
    """
    n = int(n)
    p = float(p)
    if n < 1:
        raise ValueError("n must be positive")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must lie in [0, 1]")
    m = n * (n - 1) // 2
    adj = [[0] * n for _ in range(n)]
    k = 0
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _u(k) < p:
                adj[i][j] = 1
                adj[j][i] = 1
                edges += 1
            k += 1
    deg = [sum(row) for row in adj]
    # connected components by breadth-first search
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        q = [s]
        size = 0
        while q:
            v = q.pop(0)
            size += 1
            for w in range(n):
                if adj[v][w] and not seen[w]:
                    seen[w] = True
                    q.append(w)
        comps.append(size)
    dens = edges / m if m > 0 else 0.0
    return RichResult(payload={
        "estimate": dens,
        "edges": edges,
        "density": dens,
        "expected_edges": m * p,
        "mean_degree": sum(deg) / n,
        "expected_degree": (n - 1) * p,
        "n_components": len(comps),
        "largest_component": max(comps),
        "giant_threshold": 1.0 / n,
        "connectivity_threshold": math.log(n) / n if n > 1 else 0.0,
        "n": n,
        "method": "Erdos-Renyi G(n,p)",
    })


def cheatsheet():
    return "erdosg: Erdos-Renyi G(n,p)"


# compact alias per ledger/NAMING.md
erdosrenyignp = erdos_renyi_gnp
