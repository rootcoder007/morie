"""Assignment, TSP, discrete location, transportation, routing, compactness and opening: brute force and closed forms."""

import itertools
import math

from morie.fn._rng import random_uniform
from morie.fn.cmpctop import polygon_compactness
from morie.fn.cwvrp import vehicle_routing_savings
from morie.fn.fclmop import flow_capturing_location
from morie.fn.lsapop import linear_assignment
from morie.fn.lscpop import set_covering_location
from morie.fn.mclpop import maximal_covering
from morie.fn.mopnop import morphological_opening
from morie.fn.pcntop import p_center
from morie.fn.pmedop import p_median
from morie.fn.trnsop import transportation_problem
from morie.fn.tspsol import travelling_salesman
from morie.fn.uflpop import facility_location

U = [float(v) for v in random_uniform(4000, seed=3, stream=0)]


def draws(k, off):
    return U[off : off + k]


def points(n, off):
    u = draws(2 * n, off)
    return [(10 * u[2 * i], 10 * u[2 * i + 1]) for i in range(n)]


def test_assignment_is_optimal_with_a_dual_certificate():
    for t in range(12):
        n, m = 5, 5 + t % 3
        u = draws(n * m, 50 * t)
        C = [[round(20 * u[i * m + j], 3) for j in range(m)] for i in range(n)]
        r = linear_assignment(C)
        brute = min(sum(C[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        assert abs(r["total"] - brute) < 1e-9 and len(set(r["cols"])) == n
        assert abs(linear_assignment([list(c) for c in zip(*C)])["total"] - brute) < 1e-9
        top = max(sum(C[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        assert abs(linear_assignment(C, maximize=True)["total"] - top) < 1e-9
        if n == m:  # reduced costs are non-negative and vanish on the matching; duals sum to the total
            uu, vv = r["row_potential"], r["col_potential"]
            assert all(C[i][j] - uu[i] - vv[j] >= -1e-9 for i in range(n) for j in range(m))
            assert abs(sum(uu) + sum(vv) - r["total"]) < 1e-9
    assert linear_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])["total"] == 5.0


def test_tsp_exact_matches_brute_force_and_two_opt_is_valid():
    for t in range(4):
        P = points(8, 700 + 20 * t)
        D = [[math.dist(p, q) for q in P] for p in P]
        brute = min(sum(D[a][b] for a, b in zip((0, *p), (*p, 0))) for p in itertools.permutations(range(1, 8)))
        ex = travelling_salesman(P)
        assert abs(ex["length"] - brute) < 1e-9 and ex["tour"][0] == 0 and ex["tour"][1] < ex["tour"][-1]
        he = travelling_salesman(P, method="heuristic")
        assert sorted(he["tour"]) == list(range(8)) and he["length"] >= brute - 1e-9
    assert travelling_salesman([(0, 0), (1, 0), (1, 1), (0, 1)])["length"] == 4.0


def test_location_models_match_brute_force():
    for t in range(4):
        P = points(9, 1000 + 30 * t)
        w = [1 + round(4 * v) for v in draws(9, 1500 + 10 * t)]
        D = [[math.dist(p, q) for q in P] for p in P]
        for p in (1, 2, 3):
            subsets = list(itertools.combinations(range(9), p))
            assert (
                abs(
                    p_median(p, P, weights=w)["objective"]
                    - min(sum(w[i] * min(D[i][j] for j in S) for i in range(9)) for S in subsets)
                )
                < 1e-9
            )
            assert (
                abs(p_center(p, P)["radius"] - min(max(min(D[i][j] for j in S) for i in range(9)) for S in subsets))
                < 1e-9
            )
            best = max(sum(w[i] for i in range(9) if any(D[i][j] <= 3.0 for j in S)) for S in subsets)
            assert maximal_covering(p, 3.0, P, weights=w)["covered_weight"] == best
            assert p_median(p, P, weights=w, max_enum=1)["objective"] >= p_median(p, P, weights=w)["objective"] - 1e-9
        k = min(
            len(S)
            for r in range(1, 10)
            for S in itertools.combinations(range(9), r)
            if all(any(D[i][j] <= 3.5 for j in S) for i in range(9))
        )
        sc = set_covering_location(3.5, P)
        assert sc["n_sites"] == k and all(any(D[i][j] <= 3.5 for j in sc["sites"]) for i in range(9))
        f = [round(2 + 6 * v, 3) for v in draws(9, 1800 + 10 * t)]
        uf = min(
            sum(f[j] for j in S) + sum(w[i] * min(D[i][j] for j in S) for i in range(9))
            for r in range(1, 10)
            for S in itertools.combinations(range(9), r)
        )
        assert abs(facility_location(f, P, weights=w)["total"] - uf) < 1e-9
        assert facility_location(f, P, weights=w, max_enum=1)["total"] >= uf - 1e-9
    assert set_covering_location(1.0, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])["sites"] == [0, 3]
    assert facility_location([1, 100, 1], [(0, 0), (1, 0), (10, 0)])["sites"] == [0, 2]


def test_transportation_is_the_lp_optimum():
    r = transportation_problem([[4, 6], [5, 3]], [10, 10], [8, 12])
    assert r["flow"] == [[8.0, 2.0], [0.0, 10.0]] and r["total_cost"] == 74.0 and r["balanced"]
    r = transportation_problem([[4, 6, 9], [5, 3, 8]], [15, 10], [8, 12, 4])  # lpSolve::lp.transport: 110
    assert r["total_cost"] == 110.0 and not r["balanced"]
    assert [sum(row) for row in r["flow"]] == [14.0, 10.0] and [sum(c) for c in zip(*r["flow"])] == [8.0, 12.0, 4.0]


def test_clarke_wright_routes_respect_capacity():
    r = vehicle_routing_savings([0, 1, 1, 1], 2, [(0, 0), (1, 0), (2, 0), (0, 5)])
    assert r["routes"] == [[1, 2], [3]] and r["length"] == 14.0
    P = [(5, 5)] + points(12, 2500)
    q = [0] + [1 + round(3 * v) for v in draws(12, 2600)]
    r = vehicle_routing_savings(q, 8, P)
    assert sorted(c for rt in r["routes"] for c in rt) == list(range(1, 13))
    assert all(ld <= 8 for ld in r["loads"])
    assert r["length"] <= sum(2 * math.dist(P[0], P[i]) for i in range(1, 13))


def test_flow_capturing_matches_brute_force():
    paths, vol = [[0, 1, 2], [3, 1, 4], [5, 6], [2, 6, 7], [7, 8]], [3, 2, 4, 1, 5]
    for p in (1, 2, 3):
        best = max(
            sum(v for path, v in zip(paths, vol) if set(path) & set(S)) for S in itertools.combinations(range(9), p)
        )
        assert flow_capturing_location(p, paths, vol)["captured_volume"] == best


def test_compactness_closed_forms():
    sq = polygon_compactness([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert (
        abs(sq["polsby_popper"] - math.pi / 4) < 1e-15
        and abs(sq["reock"] - 2 / math.pi) < 1e-15
        and sq["convex_hull"] == 1.0
    )
    assert abs(sq["schwartzberg"] - math.sqrt(math.pi) / 2) < 1e-15
    L = polygon_compactness([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    assert abs(L["convex_hull"] - 3 / 3.5) < 1e-15 and abs(L["reock"] - 3 / (2 * math.pi)) < 1e-12
    ngon = polygon_compactness([(math.cos(2 * math.pi * t / 64), math.sin(2 * math.pi * t / 64)) for t in range(64)])
    exact = (64 / 2 * math.sin(2 * math.pi / 64)) * 4 * math.pi / (64 * 2 * math.sin(math.pi / 64)) ** 2
    assert abs(ngon["polsby_popper"] - exact) < 1e-12 and abs(ngon["enclosing_circle"][1] - 1) < 1e-12


def test_opening_is_anti_extensive_idempotent_and_removes_small_features():
    img = [[round(9 * v) for v in draws(7, 3000 + 7 * r)] for r in range(6)]
    o = morphological_opening(img)["opened"]
    assert all(o[y][x] <= img[y][x] for y in range(6) for x in range(7))
    assert morphological_opening(o)["opened"] == o
    b = [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    r = morphological_opening(b, [[1, 1], [1, 1]], (0, 0))["opened"]
    assert r == [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]


def test_heuristics_follow_their_published_moves():
    P = points(14, 2000)
    D = [[math.dist(p, q) for q in P] for p in P]
    t = travelling_salesman(P, method="heuristic")["tour"]
    assert t == [0, 1, 7, 11, 5, 10, 4, 9, 12, 3, 8, 6, 13, 2]
    n = len(t)  # a 2-opt local optimum: no pair of edges can be exchanged for a shorter tour
    for a in range(n - 1):
        for b in range(a + 2, n if a > 0 else n - 1):
            i, i1, j, j1 = t[a], t[a + 1], t[b], t[(b + 1) % n]
            assert D[i][j] + D[i1][j1] >= D[i][i1] + D[j][j1] - 1e-12
    pins = {0: ([0, 1, 5, 7, 8], [6, 7, 8]), 1: ([0, 1, 4, 6, 7, 8], [1, 6, 8]), 2: ([0, 1, 3, 4, 6, 8], [2, 4, 5])}
    for k, (uf, pm) in pins.items():
        P = points(9, 1000 + 30 * k)
        w = [1 + round(4 * v) for v in draws(9, 1500 + 10 * k)]
        f = [round(2 + 6 * v, 3) for v in draws(9, 1800 + 10 * k)]
        assert facility_location(f, P, weights=w, max_enum=1)["sites"] == uf
        assert p_median(3, P, weights=w, max_enum=1)["sites"] == pm
