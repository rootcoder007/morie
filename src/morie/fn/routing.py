# morie.fn -- function file (rootcoder007/morie)
"""Arc and vehicle routing: the Chinese postman tour, multi-depot and periodic routing on Clarke-Wright
savings, Solomon's insertion heuristic with time windows, pickup-and-delivery (dial-a-ride) insertion,
and the expected cost of an a-priori route with stochastic demands."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .cwvrp import vehicle_routing_savings

__all__ = [
    "chinese_postman",
    "multi_depot_vrp",
    "solomon_vrptw",
    "pickup_delivery_insertion",
    "periodic_vrp",
    "stochastic_route_cost",
]


def _dist(P):
    return [[math.sqrt(ssum((a - b) ** 2 for a, b in zip(p, q))) for q in P] for p in P]


def chinese_postman(n: int, edges, *, start: int = 0) -> RichResult:
    r"""Shortest closed walk traversing every edge of a connected undirected graph (Chinese postman problem).

    Odd-degree vertices are paired by a minimum-weight perfect matching on
    shortest-path distances (exact dynamic programming over subsets); the
    matched shortest paths are duplicated, making every degree even, and an
    Euler circuit is traced by Hierholzer's algorithm (lowest neighbour and
    edge first). ``edges`` are ``(u, v, w)`` with vertices ``0 .. n-1``.

    References
    ----------
    Edmonds, J. and Johnson, E. L. (1973). Matching, Euler tours and the
    Chinese postman. *Mathematical Programming*, 5, 88-124.
    Kwan, M.-K. (1962). Graphic programming using odd or even points.
    *Chinese Mathematics*, 1, 273-277.

    Examples
    --------
    >>> r = chinese_postman(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1), (0, 2, 2)])
    >>> r.length, r.circuit[0] == r.circuit[-1]
    (8.0, True)
    """
    E = [(int(u), int(v), float(w)) for u, v, w in edges]
    D = [[0.0 if i == j else math.inf for j in range(n)] for i in range(n)]
    nxt = [[j for j in range(n)] for _ in range(n)]
    for u, v, w in E:
        if w < D[u][v]:
            D[u][v] = D[v][u] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i][k] + D[k][j] < D[i][j]:
                    D[i][j] = D[i][k] + D[k][j]
                    nxt[i][j] = nxt[i][k]
    deg = [0] * n
    for u, v, _ in E:
        deg[u] += 1
        deg[v] += 1
    odd = [i for i in range(n) if deg[i] % 2]
    m = len(odd)
    memo = {0: (0.0, [])}

    def best(mask):
        if mask in memo:
            return memo[mask]
        i = next(b for b in range(m) if mask >> b & 1)
        res = (math.inf, [])
        for j in range(i + 1, m):
            if mask >> j & 1:
                c, pr = best(mask & ~(1 << i) & ~(1 << j))
                c += D[odd[i]][odd[j]]
                if c < res[0]:
                    res = (c, [(odd[i], odd[j])] + pr)
        memo[mask] = res
        return res

    extra, pairs = best((1 << m) - 1)
    multi = [(u, v, w) for u, v, w in E]
    for a, b in pairs:
        x = a
        while x != b:
            y = nxt[x][b]
            w = min(ww for uu, vv, ww in E if {uu, vv} == {x, y})
            multi.append((x, y, w))
            x = y
    adj = [[] for _ in range(n)]
    for k, (u, v, _) in enumerate(multi):
        adj[u].append((v, k))
        adj[v].append((u, k))
    for a in adj:
        a.sort()
    used = [False] * len(multi)
    ptr = [0] * n
    stack, circ = [start], []
    while stack:
        v = stack[-1]
        while ptr[v] < len(adj[v]) and used[adj[v][ptr[v]][1]]:
            ptr[v] += 1
        if ptr[v] == len(adj[v]):
            circ.append(stack.pop())
        else:
            w, k = adj[v][ptr[v]]
            used[k] = True
            stack.append(w)
    circ.reverse()
    return RichResult(
        payload={"length": ssum(w for _, _, w in multi), "circuit": circ, "matching": pairs, "added_length": extra}
    )


def multi_depot_vrp(depots, customers, demand, capacity: float) -> RichResult:
    r"""Multi-depot vehicle routing: cluster customers to their nearest depot, then Clarke-Wright savings per depot.

    The cluster-first route-second heuristic of Gillett and Johnson (1976);
    routes are returned per depot as lists of customer indices (0-based).

    References
    ----------
    Gillett, B. E. and Johnson, J. G. (1976). Multi-terminal vehicle-dispatch
    algorithm. *Omega*, 4, 711-718.
    Clarke, G. and Wright, J. W. (1964). *Operations Research*, 12, 568-581.

    Examples
    --------
    >>> r = multi_depot_vrp([(0, 0), (10, 0)], [(1, 0), (2, 0), (9, 0), (8, 1)], [1, 1, 1, 1], 5)
    >>> r.routes
    [[[0, 1]], [[2, 3]]]
    """
    Dp = [(float(a), float(b)) for a, b in depots]
    C = [(float(a), float(b)) for a, b in customers]
    q = [float(v) for v in demand]
    assign = [min(range(len(Dp)), key=lambda d: (math.dist(c, Dp[d]), d)) for c in C]
    routes, total = [], 0.0
    for d in range(len(Dp)):
        idx = [i for i in range(len(C)) if assign[i] == d]
        if not idx:
            routes.append([])
            continue
        r = vehicle_routing_savings([0.0] + [q[i] for i in idx], capacity, [Dp[d]] + [C[i] for i in idx])
        routes.append([[idx[k - 1] for k in rt] for rt in r["routes"]])
        total += r["length"]
    return RichResult(payload={"routes": routes, "assignment": assign, "length": total})


def _schedule(route, D, ready, due, service):
    t, sched = 0.0, []
    prev = 0
    for u in route:
        t = max(ready[u], t + D[prev][u])
        if t > due[u] + 1e-9:
            return None
        sched.append(t)
        t += service[u]
        prev = u
    if t + D[prev][0] > due[0] + 1e-9:
        return None
    return sched


def solomon_vrptw(points, demand, ready, due, service, capacity: float) -> RichResult:
    r"""Vehicle routing with time windows by Solomon's I1 sequential insertion heuristic.

    Node 0 is the depot. Each route is seeded with the unrouted customer
    farthest from the depot; every unrouted customer ``u`` gets its cheapest
    feasible insertion ``c1 = d_iu + d_uj - d_ij`` (time windows, including
    the depot's closing time, and capacity checked by rescheduling), and the
    customer maximising ``c2 = d_0u - c1`` is inserted (``mu = lambda = 1``,
    ``alpha_1 = 1``). Travel times equal distances; service starts at
    ``max(ready, arrival)``.

    References
    ----------
    Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling
    problems with time window constraints. *Operations Research*, 35, 254-265.

    Examples
    --------
    >>> P = [(0, 0), (1, 0), (2, 0), (0, 3)]
    >>> solomon_vrptw(P, [0, 1, 1, 1], [0, 0, 0, 0], [100, 10, 10, 10], [0, 0, 0, 0], 5).routes
    [[1, 2, 3]]
    """
    P = [tuple(float(v) for v in p) for p in points]
    D = _dist(P)
    q = [float(v) for v in demand]
    e, lt, s = [float(v) for v in ready], [float(v) for v in due], [float(v) for v in service]
    n = len(P)
    un = list(range(1, n))
    routes = []
    while un:
        seed = max(un, key=lambda u: (D[0][u], -u))
        if _schedule([seed], D, e, lt, s) is None or q[seed] > capacity:
            raise ValueError("customer %d cannot be served on its own" % seed)
        route = [seed]
        un.remove(seed)
        while True:
            best = None
            load = ssum(q[v] for v in route)
            for u in un:
                if load + q[u] > capacity + 1e-12:
                    continue
                bu = None
                for pos in range(len(route) + 1):
                    i = route[pos - 1] if pos > 0 else 0
                    j = route[pos] if pos < len(route) else 0
                    cand = route[:pos] + [u] + route[pos:]
                    if _schedule(cand, D, e, lt, s) is None:
                        continue
                    c1 = D[i][u] + D[u][j] - D[i][j]
                    if bu is None or c1 < bu[0] - 1e-12:
                        bu = (c1, pos)
                if bu is not None:
                    c2 = D[0][u] - bu[0]
                    if best is None or c2 > best[0] + 1e-12:
                        best = (c2, u, bu[1])
            if best is None:
                break
            _, u, pos = best
            route.insert(pos, u)
            un.remove(u)
        routes.append(route)
    length = ssum(D[0][r[0]] + ssum(D[r[k]][r[k + 1]] for k in range(len(r) - 1)) + D[r[-1]][0] for r in routes)
    return RichResult(
        payload={"routes": routes, "length": length, "schedules": [_schedule(r, D, e, lt, s) for r in routes]}
    )


def pickup_delivery_insertion(points, requests, capacity: float, *, max_ride: float | None = None) -> RichResult:
    r"""Pickup-and-delivery routing (dial-a-ride with ``max_ride``) by cheapest feasible insertion.

    Requests ``(pickup, delivery, load)`` (node indices, depot 0) are inserted
    in turn at the pickup and delivery positions of least added length over
    all routes, keeping the pickup before its delivery, the on-board load
    within ``capacity`` and, for dial-a-ride, each ride (travel from pickup to
    delivery along the route) within ``max_ride``; a request fitting nowhere
    opens a new route.

    References
    ----------
    Savelsbergh, M. W. P. and Sol, M. (1995). The general pickup and delivery
    problem. *Transportation Science*, 29, 17-29.
    Jaw, J.-J., Odoni, A. R., Psaraftis, H. N. and Wilson, N. H. M. (1986). A
    heuristic algorithm for the multi-vehicle advance request dial-a-ride
    problem with time windows. *Transportation Research B*, 20, 243-257.

    Examples
    --------
    >>> P = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
    >>> pickup_delivery_insertion(P, [(1, 2, 1), (3, 4, 1)], 2).routes
    [[1, 3, 4, 2]]
    """
    Pt = [tuple(float(v) for v in p) for p in points]
    D = _dist(Pt)
    reqs = [(int(p), int(d), float(w)) for p, d, w in requests]
    load_of = {}
    for p, d, w in reqs:
        load_of[p] = w
        load_of[d] = -w

    def feasible(r):
        onb = 0.0
        for v in r:
            onb += load_of[v]
            if onb > capacity + 1e-12:
                return False
        if max_ride is not None:
            pos = {v: k for k, v in enumerate(r)}
            for p, d, _ in reqs:
                if p in pos and d in pos:
                    ride = ssum(D[r[k]][r[k + 1]] for k in range(pos[p], pos[d]))
                    if ride > max_ride + 1e-12:
                        return False
        return True

    def length(r):
        if not r:
            return 0.0
        return D[0][r[0]] + ssum(D[r[k]][r[k + 1]] for k in range(len(r) - 1)) + D[r[-1]][0]

    routes = []
    for p, d, _w in reqs:
        best = None
        for ri, r in enumerate(routes):
            base = length(r)
            for i in range(len(r) + 1):
                for j in range(i + 1, len(r) + 2):
                    cand = r[:i] + [p] + r[i:]
                    cand = cand[:j] + [d] + cand[j:]
                    if not feasible(cand):
                        continue
                    add = length(cand) - base
                    if best is None or add < best[0] - 1e-12:
                        best = (add, ri, cand)
        if best is None:
            if not feasible([p, d]):
                raise ValueError("request (%d, %d) is infeasible on its own" % (p, d))
            routes.append([p, d])
        else:
            routes[best[1]] = best[2]
    return RichResult(payload={"routes": routes, "length": ssum(length(r) for r in routes)})


def periodic_vrp(points, demand, frequency, horizon: int, capacity: float) -> RichResult:
    r"""Periodic vehicle routing: assign evenly spaced visit patterns, then route each day by Clarke-Wright savings.

    Customer ``i`` needing ``f_i`` visits over ``horizon`` days (``horizon``
    divisible by ``f_i``) is visited on days ``o, o + H/f_i, ...``; customers
    are assigned in decreasing ``demand * frequency`` to the offset ``o`` that
    minimises the largest daily load it touches (the Beltrami-Bodin
    assign-then-route scheme), and each day is routed from depot 0.

    References
    ----------
    Beltrami, E. J. and Bodin, L. D. (1974). Networks and vehicle routing for
    municipal waste collection. *Networks*, 4, 65-94.
    Christofides, N. and Beasley, J. E. (1984). The period routing problem.
    *Networks*, 14, 237-256.

    Examples
    --------
    >>> P = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    >>> r = periodic_vrp(P, [0, 2, 2, 2, 2], [0, 2, 1, 1, 1], 2, 10)
    >>> [sorted(v for rt in day for v in rt) for day in r.routes]
    [[1, 2, 4], [1, 3]]
    """
    Pt = [tuple(float(v) for v in p) for p in points]
    q = [float(v) for v in demand]
    f = [int(v) for v in frequency]
    n = len(Pt)
    dayload = [0.0] * horizon
    days_of = {}
    for i in sorted(range(1, n), key=lambda i: (-q[i] * f[i], i)):
        if f[i] <= 0 or horizon % f[i]:
            raise ValueError("frequencies must be positive divisors of the horizon")
        step = horizon // f[i]
        o = min(range(step), key=lambda o: (max(dayload[d] for d in range(o, horizon, step)), o))
        days_of[i] = list(range(o, horizon, step))
        for d in days_of[i]:
            dayload[d] += q[i]
    routes, total = [], 0.0
    for d in range(horizon):
        idx = [i for i in range(1, n) if d in days_of[i]]
        if not idx:
            routes.append([])
            continue
        r = vehicle_routing_savings([0.0] + [q[i] for i in idx], capacity, [Pt[0]] + [Pt[i] for i in idx])
        routes.append([[idx[k - 1] for k in rt] for rt in r["routes"]])
        total += r["length"]
    return RichResult(payload={"routes": routes, "days": days_of, "length": total, "daily_load": dayload})


def stochastic_route_cost(route, points, demand_pmfs, capacity: int) -> RichResult:
    r"""Expected length of an a-priori route with stochastic integer demands and restocking on failure.

    The vehicle visits ``route`` (customers, depot 0 at both ends) in order;
    when the cumulative demand first exceeds a multiple of ``capacity`` at a
    customer it returns to the depot to refill and comes back (detour
    ``2 d_0i``). With independent demands (``demand_pmfs[i]`` maps values to
    probabilities, for customer ``route[i]``) the expected length is the
    route length plus ``sum_i sum_{k >= 1} P(S_{i-1} <= kQ < S_i) 2 d_0i`` with
    ``S_i`` the cumulative demand (exact convolution).

    References
    ----------
    Dror, M., Laporte, G. and Trudeau, P. (1989). Vehicle routing with
    stochastic demands: properties and solution frameworks. *Transportation
    Science*, 23, 166-176.
    Bertsimas, D. J. (1992). A vehicle routing problem with stochastic demand.
    *Operations Research*, 40, 574-585.

    Examples
    --------
    >>> P = [(0, 0), (3, 0), (3, 4)]
    >>> r = stochastic_route_cost([1, 2], P, [{1: 0.5, 3: 0.5}, {1: 0.5, 3: 0.5}], 4)
    >>> r.expected_length, r.failure_probability
    (14.5, [0.0, 0.25])
    """
    Pt = [tuple(float(v) for v in p) for p in points]
    D = _dist(Pt)
    Q = int(capacity)
    base = D[0][route[0]] + ssum(D[route[k]][route[k + 1]] for k in range(len(route) - 1)) + D[route[-1]][0]
    S = {0: 1.0}
    extra, fails = 0.0, []
    for i, c in enumerate(route):
        pmf = {int(k): float(v) for k, v in demand_pmfs[i].items()}
        new = {}
        for s, ps in S.items():
            for x, px in pmf.items():
                new[s + x] = new.get(s + x, 0.0) + ps * px
        pf = 0.0
        for s, ps in S.items():
            m = max(Q, -(-s // Q) * Q)  # smallest positive multiple of Q not below s
            for x, px in pmf.items():
                if m < s + x:
                    pf += ps * px
        fails.append(pf)
        extra += pf * 2 * D[0][c]
        S = new
    return RichResult(payload={"expected_length": base + extra, "route_length": base, "failure_probability": fails})


def cheatsheet() -> str:
    return (
        "chinese_postman / multi_depot_vrp / solomon_vrptw / pickup_delivery_insertion / periodic_vrp / "
        "stochastic_route_cost -> arc and vehicle routing."
    )
