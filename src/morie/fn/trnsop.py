"""Transportation (allocation) problem by successive shortest paths.

Hitchcock, F. L. (1941). The distribution of a product from several sources to numerous
localities. Journal of Mathematics and Physics 20, 224-230. Solved as a min-cost flow with
successive shortest augmenting paths (Busacker and Gowen 1960) under Johnson potentials.
"""

import math

from ._richresult import RichResult

__all__ = ["transportation_problem"]


def transportation_problem(cost, supply, demand):
    r"""Minimise sum c_ij x_ij subject to row sums = supply, column sums = demand, x >= 0.

    Unbalanced problems get a zero-cost dummy source or sink. Each augmentation sends
    the bottleneck amount along a shortest source-to-sink path in the residual network
    (Bellman-Ford, which handles the negative residual costs), so the flow stays optimal
    for its value; the final flow is an optimal transportation plan.

    Parameters
    ----------
    cost : matrix (sources x sinks)
    supply, demand : sequences of non-negative amounts

    Returns
    -------
    RichResult
        Keys: flow (sources x sinks, dummy rows or columns dropped), total_cost, balanced.

    References
    ----------
    Hitchcock, F. L. (1941). Journal of Mathematics and Physics 20, 224-230.

    Examples
    --------
    >>> transportation_problem([[4, 6], [5, 3]], [10, 10], [8, 12])["total_cost"]
    74.0
    """
    C = [[float(v) for v in r] for r in cost]
    s = [float(v) for v in supply]
    d = [float(v) for v in demand]
    if len(C) != len(s) or any(len(r) != len(d) for r in C) or min(s + d) < 0:
        raise ValueError("cost must be len(supply) x len(demand) with non-negative amounts")
    ns, nd = len(s), len(d)
    S, T = sum(s), sum(d)
    bal = abs(S - T) <= 1e-9 * max(1.0, S, T)
    if not bal and S > T:
        C = [r + [0.0] for r in C]
        d = d + [S - T]
    elif not bal:
        C = C + [[0.0] * nd]
        s = s + [T - S]
    a, b = len(s), len(d)
    N = a + b + 2
    src, snk = a + b, a + b + 1
    cap, cst, adj = {}, {}, [[] for _ in range(N)]

    def add(u, v, c, w):
        cap[(u, v)], cst[(u, v)] = c, w
        cap[(v, u)], cst[(v, u)] = 0.0, -w
        adj[u].append(v)
        adj[v].append(u)

    for i in range(a):
        add(src, i, s[i], 0.0)
        for j in range(b):
            add(i, a + j, math.inf, C[i][j])
    for j in range(b):
        add(a + j, snk, d[j], 0.0)
    need = sum(s)
    sent = 0.0
    while need - sent > 1e-12 * max(1.0, need):
        dist, prev = [math.inf] * N, [-1] * N
        dist[src] = 0.0
        for _ in range(N - 1):
            ch = False
            for u in range(N):
                if dist[u] == math.inf:
                    continue
                for v in adj[u]:
                    if cap[(u, v)] > 1e-15 and dist[u] + cst[(u, v)] < dist[v] - 1e-15:
                        dist[v], prev[v], ch = dist[u] + cst[(u, v)], u, True
            if not ch:
                break
        if dist[snk] == math.inf:
            break
        f, v = math.inf, snk
        while v != src:
            f = min(f, cap[(prev[v], v)])
            v = prev[v]
        v = snk
        while v != src:
            u = prev[v]
            cap[(u, v)] -= f
            cap[(v, u)] += f
            v = u
        sent += f
    X = [[cap[(a + j, i)] for j in range(nd)] for i in range(ns)]
    tot = 0.0
    for i in range(ns):
        for j in range(nd):
            tot += X[i][j] * float(cost[i][j])
    return RichResult(
        title="Transportation problem",
        summary_lines=[("total cost", tot)],
        payload={"flow": X, "total_cost": tot, "balanced": bal},
    )


def cheatsheet():
    return "trnsop: transportation problem by successive shortest paths"
