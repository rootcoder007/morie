import itertools
import math

from morie.fn._rng import random_uniform
from morie.fn.routing import (
    chinese_postman,
    multi_depot_vrp,
    periodic_vrp,
    pickup_delivery_insertion,
    solomon_vrptw,
    stochastic_route_cost,
)

E = [
    (0, 1, 2.0),
    (1, 2, 1.5),
    (2, 3, 2.2),
    (3, 0, 1.1),
    (0, 2, 2.9),
    (1, 4, 0.7),
    (4, 5, 1.3),
    (5, 2, 0.4),
    (3, 5, 3.1),
]


def _pts(n, seed=12):
    u = [float(v) for v in random_uniform(2 * n, seed=seed)]
    return [(10 * u[2 * i], 10 * u[2 * i + 1]) for i in range(n)]


def test_postman_covers_every_edge():
    r = chinese_postman(6, E)
    c = r.circuit
    assert c[0] == c[-1] == 0
    walked = [frozenset((c[k], c[k + 1])) for k in range(len(c) - 1)]
    assert all(frozenset((u, v)) in walked for u, v, _ in E)
    wmin = {}
    for u, v, w in E:
        wmin[frozenset((u, v))] = min(w, wmin.get(frozenset((u, v)), math.inf))
    assert abs(sum(wmin[e] for e in walked) - r.length) < 1e-12
    # odd vertices 0, 1, 3, 5: brute-force the three pairings on shortest paths
    import itertools as it

    n = 6
    D = [[0 if i == j else math.inf for j in range(n)] for i in range(n)]
    for u, v, w in E:
        D[u][v] = D[v][u] = min(D[u][v], w)
    for k, i, j in it.product(range(n), repeat=3):
        D[i][j] = min(D[i][j], D[i][k] + D[k][j])
    pairings = [((0, 1), (3, 5)), ((0, 3), (1, 5)), ((0, 5), (1, 3))]
    best = min(D[a][b] + D[c_][d] for (a, b), (c_, d) in pairings)
    assert abs(r.length - (sum(w for _, _, w in E) + best)) < 1e-12


def test_multi_depot_and_periodic():
    C = _pts(16)
    dem = [1 + i % 3 for i in range(16)]
    dep = [(2, 2), (8, 8), (2, 8)]
    r = multi_depot_vrp(dep, C, dem, 6)
    served = sorted(c for rs in r.routes for rt in rs for c in rt)
    assert served == list(range(16))
    for d, rs in enumerate(r.routes):
        for rt in rs:
            assert sum(dem[c] for c in rt) <= 6
            assert all(min(range(3), key=lambda k: (math.dist(C[c], dep[k]), k)) == d for c in rt)
    P = [(5, 5)] + C[:10]
    f = [0, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1]
    p = periodic_vrp(P, [0] + dem[:10], f, 4, 7)
    for i in range(1, 11):
        visits = [d for d, day in enumerate(p.routes) if any(i in rt for rt in day)]
        assert len(visits) == f[i] and visits == p.days[i]


def test_solomon_schedule_is_feasible():
    C = _pts(10)
    u = [float(v) for v in random_uniform(200, seed=12)]
    P = [(5, 5)] + C
    dem = [0] + [1 + i % 3 for i in range(10)]
    rd = [0] + [10 * u[40 + i] for i in range(10)]
    du = [200] + [rd[i + 1] + 15 + 10 * u[60 + i] for i in range(10)]
    r = solomon_vrptw(P, dem, rd, du, [0] + [1.0] * 10, 8)
    assert sorted(c for rt in r.routes for c in rt) == list(range(1, 11))
    for rt, sc in zip(r.routes, r.schedules):
        t, prev = 0.0, 0
        for c, s in zip(rt, sc):
            t = max(rd[c], t + math.dist(P[prev], P[c]))
            assert abs(t - s) < 1e-12 and t <= du[c]
            t += 1.0
            prev = c
        assert sum(dem[c] for c in rt) <= 8


def test_pickup_delivery_constraints():
    P = [(5, 5)] + _pts(10)
    req = [(1, 2, 1), (3, 4, 2), (5, 6, 1), (7, 8, 1), (9, 10, 2)]
    r = pickup_delivery_insertion(P, req, 3, max_ride=13.0)
    for rt in r.routes:
        load = 0
        for v in rt:
            load += next(w if v == p else -w for p, d, w in req if v in (p, d))
            assert load <= 3
        for p, d, _ in req:
            if p in rt:
                i, j = rt.index(p), rt.index(d)
                assert i < j and sum(math.dist(P[rt[k]], P[rt[k + 1]]) for k in range(i, j)) <= 13.0


def test_stochastic_cost_by_enumeration():
    P = [(5, 5)] + _pts(10)
    route, Q = [3, 1, 4, 2, 5], 6
    pmf = {1: 0.3, 2: 0.4, 4: 0.3}
    r = stochastic_route_cost(route, P, [pmf] * 5, Q)
    exp = 0.0
    for xs in itertools.product(pmf, repeat=5):
        pr = math.prod(pmf[x] for x in xs)
        length, load, prev = 0.0, Q, 0
        for c, x in zip(route, xs):
            length += math.dist(P[prev], P[c])
            if x > load:
                length += 2 * math.dist(P[0], P[c])
                load = Q - (x - load)
            else:
                load -= x
            prev = c
        exp += pr * (length + math.dist(P[prev], P[0]))
    assert abs(r.expected_length - exp) < 1e-12
