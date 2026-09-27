"""Capacitated vehicle routing by the Clarke-Wright savings algorithm.

Clarke, G. and Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number
of delivery points. Operations Research 12, 568-581.
"""

import math

from ._richresult import RichResult

__all__ = ["vehicle_routing_savings"]


def vehicle_routing_savings(demand, capacity, points=None, dist=None):
    r"""Routes from depot 0 by the parallel Clarke-Wright savings heuristic.

    Start with one out-and-back route per customer; consider pairs in decreasing savings
    s_ij = d_0i + d_0j - d_ij (ties by (i, j)); merge the routes of i and j when they
    differ, i and j are route ends (each adjacent to the depot) and the combined load
    fits the capacity.

    Parameters
    ----------
    demand : sequence
        Demand per node; entry 0 (the depot) is ignored.
    capacity : float
    points : coordinates, optional
    dist : matrix, optional

    Returns
    -------
    RichResult
        Keys: routes (customer lists, depot omitted), loads, length (total), route_lengths.

    References
    ----------
    Clarke, G. and Wright, J. W. (1964). Operations Research 12, 568-581.

    Examples
    --------
    >>> vehicle_routing_savings([0, 1, 1, 1], 2, [(0, 0), (1, 0), (2, 0), (0, 5)])["routes"]
    [[1, 2], [3]]
    """
    if (points is None) == (dist is None):
        raise ValueError("give exactly one of points and dist")
    if dist is not None:
        D = [[float(v) for v in r] for r in dist]
    else:
        P = [[float(v) for v in p] for p in points]
        D = [[math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q))) for q in P] for p in P]
    n = len(D)
    q = [float(v) for v in demand]
    Q = float(capacity)
    if len(q) != n or any(q[i] > Q for i in range(1, n)):
        raise ValueError("one demand per node, each at most the capacity")
    route = {i: [i] for i in range(1, n)}
    load = {i: q[i] for i in range(1, n)}
    of = {i: i for i in range(1, n)}
    sav = sorted(
        ((D[0][i] + D[0][j] - D[i][j], i, j) for i in range(1, n) for j in range(i + 1, n)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    for s, i, j in sav:
        if s <= 0:
            break
        ri, rj = of[i], of[j]
        if ri == rj or load[ri] + load[rj] > Q + 1e-12:
            continue
        A, B = route[ri], route[rj]
        if A[-1] == i and B[0] == j:
            new = A + B
        elif A[0] == i and B[-1] == j:
            new = B + A
        elif A[0] == i and B[0] == j:
            new = A[::-1] + B
        elif A[-1] == i and B[-1] == j:
            new = A + B[::-1]
        else:
            continue
        route[ri], load[ri] = new, load[ri] + load[rj]
        del route[rj], load[rj]
        for k in new:
            of[k] = ri
    routes = []
    for r in route.values():
        routes.append(r if r[0] <= r[-1] else r[::-1])
    routes.sort()
    lens = []
    for r in routes:
        L = D[0][r[0]] + D[r[-1]][0]
        for a, b in zip(r, r[1:]):
            L += D[a][b]
        lens.append(L)
    tot = 0.0
    for v in lens:
        tot += v
    loads = []
    for r in routes:
        s = 0.0
        for k in r:
            s += q[k]
        loads.append(s)
    return RichResult(
        title="Vehicle routing (Clarke-Wright)",
        summary_lines=[("routes", len(routes)), ("length", tot)],
        payload={"routes": routes, "loads": loads, "length": tot, "route_lengths": lens},
    )


def cheatsheet():
    return "cwvrp: capacitated vehicle routing by Clarke-Wright savings"
