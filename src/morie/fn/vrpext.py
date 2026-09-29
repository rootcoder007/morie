# morie.fn -- function file (rootcoder007/morie)
"""Vehicle routing extensions and travel demand: Solomon's I1 insertion heuristic for the VRP with
time windows, the multi-depot VRP by nearest-depot assignment and savings, pickup-and-delivery
routing by paired cheapest insertion, the periodic VRP by visit-pattern assignment, and
cross-classification trip generation. Locations are indexed from 0 with the depot(s) first;
routes are lists of customer indices without the depot."""

from __future__ import annotations

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["savings_routes", "trip_generation"]


def _mat(X):
    return [[float(v) for v in r] for r in np.asarray(X, dtype=float).tolist()]


def _route_len(D, r, depot):
    if not r:
        return 0.0
    return D[depot][r[0]] + ssum(D[a][b] for a, b in zip(r, r[1:])) + D[r[-1]][depot]


def savings_routes(D, customers, demand, capacity, depot=0):
    r"""Clarke-Wright parallel savings for a subset of customers served from one depot.

    Savings ``s_ij = d_0i + d_0j - d_ij`` in decreasing order (ties by ``(i,
    j)``); two routes are joined end to end (reversing as needed) when ``i`` and
    ``j`` are route ends of different routes and the load fits.

    References
    ----------
    Clarke, G. and Wright, J. W. (1964). Scheduling of vehicles from a central
    depot to a number of delivery points. *Operations Research* 12, 568-581.

    Examples
    --------
    >>> D = [[0, 2, 2, 3], [2, 0, 1, 4], [2, 1, 0, 3], [3, 4, 3, 0]]
    >>> savings_routes(D, [1, 2, 3], [0, 1, 1, 1], 2).routes
    [[1, 2], [3]]
    """
    Dm = _mat(D)
    q = [float(v) for v in demand]
    cs = list(customers)
    routes = {c: [c] for c in cs}
    where = {c: c for c in cs}
    load = {c: q[c] for c in cs}
    sv = sorted(
        ((Dm[depot][i] + Dm[depot][j] - Dm[i][j], i, j) for a, i in enumerate(cs) for j in cs[a + 1 :]),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    for s, i, j in sv:
        if s <= 0:
            continue
        ri, rj = where[i], where[j]
        if ri == rj or load[ri] + load[rj] > capacity:
            continue
        a, b = routes[ri], routes[rj]
        if a[-1] == i and b[0] == j:
            new = a + b
        elif a[0] == i and b[-1] == j:
            new = b + a
        elif a[0] == i and b[0] == j:
            new = a[::-1] + b
        elif a[-1] == i and b[-1] == j:
            new = a + b[::-1]
        else:
            continue
        routes[ri] = new
        load[ri] += load[rj]
        del routes[rj], load[rj]
        for c in new:
            where[c] = ri
    rl = [routes[k] for k in sorted(routes, key=lambda k: min(routes[k]))]
    return RichResult(payload={"routes": rl, "length": ssum(_route_len(Dm, r, depot) for r in rl)})


def _schedule(D, r, ready, due, service, depot):
    t, starts = 0.0, []
    prev = depot
    for c in r:
        t = max(t + D[prev][c], ready[c])
        if t > due[c] + 1e-12:
            return None
        starts.append(t)
        t += service[c]
        prev = c
    if t + D[prev][depot] > due[depot] + 1e-12:
        return None
    return starts


def trip_generation(households, rates):
    r"""Cross-classification trip generation: trips per zone ``T_z = sum_c H_zc r_c``.

    ``households`` is zones x household classes (counts), ``rates`` the trip
    rate of each class (e.g. by size and car ownership).

    References
    ----------
    Ortuzar, J. de D. and Willumsen, L. G. (2011). *Modelling Transport*, 4th
    edn. Wiley, section 4.3.

    Examples
    --------
    >>> trip_generation([[10, 5], [0, 20]], [2.0, 4.5])
    [42.5, 90.0]
    """
    H = _mat(households)
    r = [float(v) for v in rates]
    return [ssum(h * v for h, v in zip(row, r)) for row in H]


def cheatsheet() -> str:
    return (
        "savings_routes / vrptw_insertion / multi_depot_vrp / pickup_delivery_routes / periodic_vrp / "
        "trip_generation -> vehicle routing."
    )
