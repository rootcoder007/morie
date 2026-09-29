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

__all__ = [
    "savings_routes",
    "vrptw_insertion",
    "multi_depot_vrp",
    "pickup_delivery_routes",
    "periodic_vrp",
    "trip_generation",
]


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


def vrptw_insertion(D, demand, capacity, ready, due, service, mu=1.0, lam=1.0, alpha1=1.0, depot=0):
    r"""Solomon (1987) I1 sequential insertion for the VRP with time windows.

    Each route is seeded with the unrouted customer farthest from the depot.
    For every unrouted customer ``u`` the best feasible position between
    ``i`` and ``j`` minimises ``c1 = alpha1 (d_iu + d_uj - mu d_ij) + (1 - alpha1)
    (b_j' - b_j)`` (push-forward of the service start at ``j``); the customer
    maximising ``c2 = lam d_0u - c1`` is inserted. A route closes when no
    feasible insertion (capacity, time windows, return to the depot by its due
    time) remains. Travel times equal ``D``.

    References
    ----------
    Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling
    problems with time window constraints. *Operations Research* 35, 254-265.

    Examples
    --------
    >>> D = [[0, 2, 3, 4], [2, 0, 2, 5], [3, 2, 0, 2], [4, 5, 2, 0]]
    >>> r = vrptw_insertion(D, [0, 1, 1, 1], 3, [0, 0, 0, 0], [100, 10, 10, 10], [0, 1, 1, 1])
    >>> r.routes
    [[1, 2, 3]]
    """
    Dm = _mat(D)
    n = len(Dm)
    q = [float(v) for v in demand]
    un = [c for c in range(n) if c != depot]
    routes = []
    while un:
        seed = max(un, key=lambda c: (Dm[depot][c], -c))
        r = [seed]
        un.remove(seed)
        if _schedule(Dm, r, ready, due, service, depot) is None:
            routes.append(r)
            continue
        while True:
            best = None
            load = ssum(q[c] for c in r)
            base = _schedule(Dm, r, ready, due, service, depot)
            for u in un:
                if load + q[u] > capacity:
                    continue
                bu = None
                for pos in range(len(r) + 1):
                    cand = r[:pos] + [u] + r[pos:]
                    st = _schedule(Dm, cand, ready, due, service, depot)
                    if st is None:
                        continue
                    i = depot if pos == 0 else r[pos - 1]
                    j = depot if pos == len(r) else r[pos]
                    c11 = Dm[i][u] + Dm[u][j] - mu * Dm[i][j]
                    c12 = 0.0 if j == depot else st[pos + 1] - base[pos]
                    c1 = alpha1 * c11 + (1 - alpha1) * c12
                    if bu is None or c1 < bu[0] - 1e-12:
                        bu = (c1, pos)
                if bu is None:
                    continue
                c2 = lam * Dm[depot][u] - bu[0]
                if best is None or c2 > best[0] + 1e-12:
                    best = (c2, u, bu[1])
            if best is None:
                break
            _, u, pos = best
            r.insert(pos, u)
            un.remove(u)
        routes.append(r)
    return RichResult(payload={"routes": routes, "length": ssum(_route_len(Dm, r, depot) for r in routes)})


def multi_depot_vrp(D, depots, demand, capacity):
    r"""Multi-depot VRP: each customer to its nearest depot, then Clarke-Wright savings per depot.

    The cluster-first route-second heuristic (Tillman 1969); ties go to the
    lower-indexed depot.

    References
    ----------
    Tillman, F. A. (1969). The multiple terminal delivery problem with
    probabilistic demands. *Transportation Science* 3, 192-204.

    Examples
    --------
    >>> D = [[0, 9, 1, 8], [9, 0, 8, 1], [1, 8, 0, 7], [8, 1, 7, 0]]
    >>> multi_depot_vrp(D, [0, 1], [0, 0, 1, 1], 5).routes
    {0: [[2]], 1: [[3]]}
    """
    Dm = _mat(D)
    dep = [int(v) for v in depots]
    cust = [c for c in range(len(Dm)) if c not in dep]
    assign = {d: [] for d in dep}
    for c in cust:
        assign[min(dep, key=lambda d: (Dm[d][c], d))].append(c)
    routes, total = {}, 0.0
    for d in dep:
        if assign[d]:
            r = savings_routes(Dm, assign[d], demand, capacity, depot=d)
            routes[d] = r.routes
            total += r.length
        else:
            routes[d] = []
    return RichResult(payload={"routes": routes, "assignment": assign, "length": total})


def pickup_delivery_routes(D, pairs, load, capacity, depot=0):
    r"""Pickup-and-delivery routing by paired cheapest insertion.

    Requests ``(p, d)`` with load ``q`` are taken in decreasing order of
    ``d_0p + d_pd + d_d0``; each is inserted into an existing route at the
    pickup and delivery positions (pickup before delivery) that add least
    length while the on-board load never exceeds ``capacity``, or starts a new
    route when no insertion is feasible.

    References
    ----------
    Savelsbergh, M. W. P. and Sol, M. (1995). The general pickup and delivery
    problem. *Transportation Science* 29, 17-29.

    Examples
    --------
    >>> D = [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]
    >>> pickup_delivery_routes(D, [(1, 3), (2, 4)], [1, 1], 2).routes
    [[1, 2, 3, 4]]
    """
    Dm = _mat(D)
    reqs = sorted(
        range(len(pairs)),
        key=lambda k: (-(Dm[depot][pairs[k][0]] + Dm[pairs[k][0]][pairs[k][1]] + Dm[pairs[k][1]][depot]), k),
    )
    qq = {}
    for k, (p, d) in enumerate(pairs):
        qq[p], qq[d] = float(load[k]), -float(load[k])
    routes = []

    def ok(r):
        s = 0.0
        for c in r:
            s += qq[c]
            if s > capacity + 1e-12:
                return False
        return True

    for k in reqs:
        p, d = pairs[k]
        best = None
        for ri, r in enumerate(routes):
            base = _route_len(Dm, r, depot)
            for a in range(len(r) + 1):
                for b in range(a, len(r) + 1):
                    cand = r[:a] + [p] + r[a:b] + [d] + r[b:]
                    if not ok(cand):
                        continue
                    inc = _route_len(Dm, cand, depot) - base
                    if best is None or inc < best[0] - 1e-12:
                        best = (inc, ri, cand)
        if best is None or float(load[k]) > capacity:
            routes.append([p, d])
        else:
            routes[best[1]] = best[2]
    return RichResult(payload={"routes": routes, "length": ssum(_route_len(Dm, r, depot) for r in routes)})


def periodic_vrp(D, demand, capacity, frequency, patterns, depot=0):
    r"""Periodic VRP: visit-pattern assignment balancing daily load, then savings routes per day.

    ``patterns[f]`` lists the admissible day sets for visit frequency ``f``.
    Customers in decreasing order of demand get the admissible pattern that
    minimises the heaviest resulting daily load (ties: first pattern); each
    day is then routed by Clarke-Wright savings (the classical two-phase
    heuristic of Beltrami and Bodin 1974).

    References
    ----------
    Beltrami, E. J. and Bodin, L. D. (1974). Networks and vehicle routing for
    municipal waste collection. *Networks* 4, 65-94.

    Examples
    --------
    >>> D = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
    >>> r = periodic_vrp(D, [0, 2, 2, 1], 5, [0, 2, 1, 1], {1: [[0], [1]], 2: [[0, 1]]})
    >>> r.days
    {0: [1, 2], 1: [1, 3]}
    """
    Dm = _mat(D)
    q = [float(v) for v in demand]
    cust = sorted((c for c in range(len(Dm)) if c != depot), key=lambda c: (-q[c], c))
    days = sorted({d for pl in patterns.values() for pat in pl for d in pat})
    loadd = {d: 0.0 for d in days}
    visits = {d: [] for d in days}
    for c in cust:
        best = None
        for pat in patterns[int(frequency[c])]:
            peak = max(loadd[d] + (q[c] if d in pat else 0.0) for d in days)
            if best is None or peak < best[0] - 1e-12:
                best = (peak, pat)
        for d in best[1]:
            loadd[d] += q[c]
            visits[d].append(c)
    out, total = {}, 0.0
    for d in days:
        r = savings_routes(Dm, sorted(visits[d]), q, capacity, depot)
        out[d] = r.routes
        total += r.length
    return RichResult(payload={"days": {d: sorted(v) for d, v in visits.items()}, "routes": out, "length": total})


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
