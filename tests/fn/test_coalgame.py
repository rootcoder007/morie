import math

from morie.fn.coalgame import (
    median_lines,
    minimal_range_coalition,
    quota_solution,
    roemer_pune,
    spatial_heart,
    weighted_game_core,
)


def test_three_party_game():
    P, w = [(0, 0), (4, 0), (0, 3)], [30, 35, 35]
    assert median_lines(P, w, 51) == [(0, 1), (0, 2), (1, 2)]
    assert weighted_game_core(P, w, 51).core == []
    assert spatial_heart(P, w, 51).vertices == [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]


def test_core_with_dominant_centre():
    P = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    assert weighted_game_core(P, [20, 20, 20, 20, 20], 51).core == [0]  # radial symmetry (Plott)
    assert weighted_game_core([(0, 0), (1, 0), (0, 1), (-1, -1)], [10, 30, 30, 30], 51).core == []
    h = spatial_heart(P, [20, 20, 20, 20, 20], 51)
    assert h.core and h.vertices == [(0.0, 0.0)]


def test_quota_equations():
    for w, q in (([1, 1, 1], 2), ([3, 2, 2, 1, 1], 5), ([4, 3, 2, 1], 6)):
        r = quota_solution(w, q)
        for S in r.minimal_winning:
            total = sum(w[i] for i in S)
            assert total >= q and all(total - w[i] < q for i in S)
        if r.consistent:
            assert all(abs(sum(r.quota[i] for i in S) - 1) < 1e-9 for S in r.minimal_winning)
    assert all(abs(v - 0.5) < 1e-12 for v in quota_solution([1, 1, 1], 2).quota)


def test_minimal_range():
    r = minimal_range_coalition([-2, -1, 0, 1, 3], [20, 15, 10, 25, 30], 51)
    assert r.coalition == [3, 4] and r.range == 2.0


def test_pune_first_order_conditions():
    a, b, mu, sig, al, be = -1.0, 1.5, 0.2, 0.6, 0.4, 0.7
    r = roemer_pune(a, b, mu, sig, al, be)

    def Phi(z):
        return 0.5 * math.erfc(-z / math.sqrt(2))

    def fa(t):
        return math.log(Phi(((t + r.s) / 2 - mu) / sig)) + (1 - al) * math.log((r.s - a) ** 2 - (t - a) ** 2)

    def fb(s):
        return math.log(1 - Phi(((r.t + s) / 2 - mu) / sig)) + (1 - be) * math.log((r.t - b) ** 2 - (s - b) ** 2)

    h = 1e-5
    assert abs((fa(r.t + h) - fa(r.t - h)) / (2 * h)) < 1e-5
    assert abs((fb(r.s + h) - fb(r.s - h)) / (2 * h)) < 1e-5
    sym = roemer_pune(-1.0, 1.0, 0.0, 0.5, 0.5, 0.5)
    assert abs(sym.t + sym.s) < 1e-7  # golden-section resolution of a flat optimum
