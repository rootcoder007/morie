"""transnet: brute-force enumerations and closed forms against the network algorithms."""

import itertools
import math

import pytest

from morie.fn.transnet import (
    clarke_wright_vrp,
    gravity_distribution,
    k_shortest_paths,
    logit_mode_shares,
    shortest_path,
    traffic_assignment,
    tsp_tour,
    webster_signal,
)

E = [(0, 1, 4.0), (0, 2, 1.0), (2, 1, 2.0), (1, 3, 1.0), (2, 3, 5.0), (3, 4, 3.0), (1, 4, 6.0), (2, 4, 9.0)]


def _all_paths(n, E, s, t):
    out = []

    def go(p, c):
        if p[-1] == t:
            out.append((c, p))
            return
        for a, b, w in E:
            if a == p[-1] and b not in p:
                go(p + [b], c + w)

    go([s], 0.0)
    return sorted(out)


def test_dijkstra_and_yen_against_enumeration():
    r = shortest_path(5, E, 0, 4)
    brute = _all_paths(5, E, 0, 4)
    assert r.distance[4] == brute[0][0] and r.path == brute[0][1]
    assert r.distance == [0.0, 3.0, 1.0, 4.0, 7.0]
    k = k_shortest_paths(5, E, 0, 4, 6)
    assert k.costs == [c for c, _ in brute[:6]]
    assert k.paths == [p for _, p in brute[:6]]
    assert k_shortest_paths(5, E, 4, 0, 3).paths == []
    with pytest.raises(ValueError):
        shortest_path(2, [(0, 1, -1.0)], 0)


def test_tsp_exact_matches_brute_force_and_heuristic_is_two_opt_local_optimum():
    pts = [(0, 0), (3, 1), (6, 0), (7, 4), (4, 6), (1, 5), (2, 2.5), (5, 3)]
    D = [[math.dist(a, b) for b in pts] for a in pts]
    best = min(
        sum(D[t[i]][t[(i + 1) % 8]] for i in range(8)) for t in ((0,) + p for p in itertools.permutations(range(1, 8)))
    )
    ex = tsp_tour(D)
    assert ex.length == pytest.approx(best, abs=1e-12)
    h = tsp_tour(D, method="heuristic")
    t, n = h.tour, 8
    assert sorted(t) == list(range(8)) and t[0] == 0
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            a, b, c, d = t[i - 1], t[i], t[j], t[(j + 1) % n]
            assert D[a][c] + D[b][d] >= D[a][b] + D[c][d] - 1e-12
    assert h.length >= ex.length - 1e-12
    asym = [[0, 1, 10], [10, 0, 1], [1, 10, 0]]
    assert tsp_tour(asym).tour == [0, 1, 2] and tsp_tour(asym).length == 3.0


def test_clarke_wright_and_gravity():
    D = [[0, 4, 4, 5, 5], [4, 0, 1, 7, 8], [4, 1, 0, 7, 7], [5, 7, 7, 0, 1], [5, 8, 7, 1, 0]]
    big = clarke_wright_vrp(D, [0, 1, 1, 1, 1], 10)
    # savings: s34 = 9, s12 = 7, then s13 = s23 = s24 = 2 (ties by (i, j)): (1, 3) joins [2, 1] to [3, 4]
    assert big.routes == [[2, 1, 3, 4]] and big.cost == 4 + 1 + 7 + 1 + 5
    one = clarke_wright_vrp(D, [0, 1, 1, 1, 1], 1)
    assert one.routes == [[1], [2], [3], [4]] and one.cost == 36.0
    with pytest.raises(ValueError):
        clarke_wright_vrp(D, [0, 3, 1, 1, 1], 2)
    g = gravity_distribution([100, 50, 30], [60, 90, 30], [[1, 2, 3], [2, 1, 2], [3, 2, 1]], 0.4)
    assert [sum(r) for r in g.trips] == pytest.approx([100, 50, 30], rel=1e-11)
    assert [sum(g.trips[i][j] for i in range(3)) for j in range(3)] == pytest.approx([60, 90, 30], rel=1e-11)
    T = g.trips
    assert T[0][0] * T[1][1] / (T[0][1] * T[1][0]) == pytest.approx(math.exp(0.4 * 2), rel=1e-10)
    pw = gravity_distribution([10, 10], [10, 10], [[1, 2], [2, 1]], 2.0, deterrence="power")
    assert pw.trips[0][0] / pw.trips[0][1] == pytest.approx(4.0, rel=1e-10)  # f = c^-2
    with pytest.raises(ValueError):
        gravity_distribution([1, 2], [1, 1], [[1, 1], [1, 1]], 1.0)


def test_logit_assignment_webster():
    r = logit_mode_shares([[1.0, 2.0, 0.5], [1000.0, 999.0, 998.0]])
    e = [math.exp(1), math.exp(2), math.exp(0.5)]
    assert r.shares[0] == pytest.approx([v / sum(e) for v in e]) and r.logsum[0] == pytest.approx(math.log(sum(e)))
    assert r.logsum[1] == pytest.approx(1000 + math.log(1 + math.exp(-1) + math.exp(-2)))
    links = [(0, 1, 10.0, 100.0), (0, 1, 15.0, 100.0), (1, 2, 5.0, 50.0), (0, 2, 30.0, 80.0)]
    ue = traffic_assignment(3, links, [(0, 2, 150.0), (0, 1, 40.0)], alpha=1.0, beta=2.0)
    t = ue.time
    # Wardrop: used paths between each pair have equal time, unused paths are not faster
    assert t[0] == pytest.approx(t[1], rel=1e-6)
    assert t[0] + t[2] == pytest.approx(t[3], rel=1e-6)
    so = traffic_assignment(3, links, [(0, 2, 150.0), (0, 1, 40.0)], method="so", alpha=1.0, beta=2.0)
    assert so.total_time <= ue.total_time + 1e-9
    mc = [tt + x * t0 * 2 * x / c**2 for tt, x, (_, _, t0, c) in zip(so.time, so.flow, links)]
    assert mc[0] == pytest.approx(mc[1], rel=1e-6) and mc[0] + mc[2] == pytest.approx(mc[3], rel=1e-6)
    aon = traffic_assignment(3, links, [(0, 2, 150.0)], method="aon")
    assert aon.flow == [150.0, 0.0, 150.0, 0.0]
    w = webster_signal([0.2, 0.1], [0.5, 0.4], 12.0)
    Y = 0.4 + 0.25
    C = (18 + 5) / (1 - Y)
    assert (w.cycle, w.green) == pytest.approx((C, [(C - 12) * 0.4 / Y, (C - 12) * 0.25 / Y]))
    lam, x, q = w.green[0] / C, 0.2 / (w.green[0] / C * 0.5), 0.2
    d0 = (
        C * (1 - lam) ** 2 / (2 * (1 - lam * x))
        + x * x / (2 * q * (1 - x))
        - 0.65 * (C / q**2) ** (1 / 3) * x ** (2 + 5 * lam)
    )
    assert w.delay[0] == pytest.approx(d0, abs=1e-12)
    with pytest.raises(ValueError):
        webster_signal([0.3, 0.3], [0.5, 0.5], 10.0)
