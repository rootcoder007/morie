"""vrpext: feasibility of every route and the defining rules of each heuristic."""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.vrpext import (
    multi_depot_vrp,
    periodic_vrp,
    pickup_delivery_routes,
    savings_routes,
    trip_generation,
    vrptw_insertion,
)

U = [float(v) for v in random_uniform(200, seed=41)]
N = 11
P = [[50.0, 50.0]] + [[100 * U[i], 100 * U[20 + i]] for i in range(1, N)]
D = [[math.dist(a, b) for b in P] for a in P]
Q = [0.0] + [1 + int(4 * U[40 + i]) for i in range(1, N)]


def _covers(routes, customers):
    flat = sorted(c for r in routes for c in r)
    return flat == sorted(customers)


def test_savings_capacity_and_cover():
    r = savings_routes(D, list(range(1, N)), Q, 10)
    assert _covers(r.routes, range(1, N)) and all(sum(Q[c] for c in x) <= 10 for x in r.routes)
    one = savings_routes(D, list(range(1, N)), Q, 1e9)
    assert len(one.routes) == 1


def test_time_windows_respected():
    ready = [0.0] + [80 * U[60 + i] for i in range(1, N)]
    due = [1000.0] + [ready[i] + 70 for i in range(1, N)]
    s = [0.0] + [4.0] * (N - 1)
    r = vrptw_insertion(D, Q, 12, ready, due, s)
    assert _covers(r.routes, range(1, N))
    for x in r.routes:
        t, prev = 0.0, 0
        assert sum(Q[c] for c in x) <= 12
        for c in x:
            t = max(t + D[prev][c], ready[c])
            assert t <= due[c] + 1e-9
            t += s[c]
            prev = c


def test_multi_depot_assignment_and_pickup_precedence():
    r = multi_depot_vrp(D, [0, 1], Q, 9)
    for d, cs in r.assignment.items():
        for c in cs:
            assert D[d][c] <= D[1 - d][c]
    pairs = [(2, 3), (4, 5), (6, 7), (8, 9)]
    load = [2, 3, 1, 2]
    pd = pickup_delivery_routes(D, pairs, load, 4)
    for x in pd.routes:
        onboard = 0
        for c in x:
            for k, (p, q) in enumerate(pairs):
                if c == p:
                    onboard += load[k]
                if c == q:
                    onboard -= load[k]
                    assert x.index(p) < x.index(q)
            assert onboard <= 4


def test_periodic_and_trip_generation():
    freq = [0] + [1 + i % 2 for i in range(1, N)]
    r = periodic_vrp(D, Q, 30, freq, {1: [[0], [1]], 2: [[0, 1]]})
    for c in range(1, N):
        assert sum(c in r.days[d] for d in (0, 1)) == freq[c]
    assert trip_generation([[1, 2], [3, 4]], [0.5, 2.0]) == pytest.approx([4.5, 9.5])
