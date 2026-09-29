"""vrpext: feasibility of every route and the defining rules of each heuristic."""

import math

from morie.fn._rng import random_uniform
from morie.fn.vrpext import savings_routes

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
