"""electgeo: measures recomputed from their definitions, plus geometric identities for compactness."""

import math

import pytest

from morie.fn.electgeo import (
    disproportionality,
    district_compactness,
    district_competitiveness,
    electoral_swing,
    malapportionment,
    partisan_gerrymander_measures,
)


def test_partisan_measures():
    a, b = [55, 58, 30, 35, 62], [45, 42, 70, 65, 38]
    r = partisan_gerrymander_measures(a, b)
    wa = 5 + 8 + 30 + 35 + 12
    wb = 45 + 42 + 20 + 15 + 38
    assert (r.wasted_a, r.wasted_b) == (wa, wb)
    assert r.efficiency_gap == pytest.approx((wa - wb) / 500, abs=1e-15)
    sh = [0.55, 0.58, 0.30, 0.35, 0.62]
    assert r.mean_median == pytest.approx(sum(sh) / 5 - 0.55, abs=1e-15)
    shift = 0.5 - sum(sh) / 5
    assert r.partisan_bias == pytest.approx(sum(s + shift > 0.5 for s in sh) / 5 - 0.5)
    sym = partisan_gerrymander_measures([60, 40], [40, 60])
    assert (sym.efficiency_gap, sym.mean_median, sym.partisan_bias) == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)
    with pytest.raises(ValueError):
        partisan_gerrymander_measures([1, 2], [1])


def test_swing_competitiveness_disproportionality():
    s = electoral_swing([0.40, 0.5], [0.45, 0.3], [0.46, 0.45], [0.41, 0.35])
    assert s.butler == pytest.approx([0.05, -0.05])
    assert s.steed == pytest.approx([0.46 / 0.87 - 0.40 / 0.85, 0.45 / 0.8 - 0.5 / 0.8])
    c = district_competitiveness([52, 70, 48, 45], [48, 30, 52, 55], threshold=0.03)
    assert c.competitive == [True, False, True, False]
    assert c.mean_margin == pytest.approx((0.04 + 0.4 + 0.04 + 0.1) / 4)
    d = disproportionality([40, 35, 25], [55, 40, 5])
    assert (d.loosemore_hanby, d.gallagher) == pytest.approx((20.0, math.sqrt(325)))
    assert (d.enp_votes, d.enp_seats) == pytest.approx((1 / 0.345, 1 / (0.3025 + 0.16 + 0.0025)))
    m = malapportionment([2, 2, 1], [100, 300, 100])
    assert m.mal == pytest.approx(0.5 * (abs(0.4 - 0.2) + abs(0.4 - 0.6) + 0.0))
    assert m.ratio == pytest.approx([2.0, 2 / 3, 1.0])


def test_compactness():
    sq = district_compactness([(0, 0), (2, 0), (2, 2), (0, 2)])
    assert (sq.polsby_popper, sq.reock, sq.convex_hull) == pytest.approx((math.pi / 4, 2 / math.pi, 1.0))
    assert sq.schwartzberg == pytest.approx(1 / math.sqrt(sq.polsby_popper))
    L = district_compactness([(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)])
    assert (L.area, L.perimeter) == (6.0, 14.0)
    assert L.convex_hull == pytest.approx(6 / (12 - 0.5 * 3 * 2))
    # minimum enclosing circle of the L: diameter is the diagonal from (0, 3) to (4, 0), i.e. 5
    assert L.reock == pytest.approx(6 / (math.pi * 6.25))
    tri = district_compactness([(0, 0), (2, 0), (1, math.sqrt(3))])
    assert tri.reock == pytest.approx(math.sqrt(3) / (math.pi * 4 / 3))
    obt = district_compactness([(0, 0), (10, 0), (5, 1)])
    assert obt.reock == pytest.approx(5 / (math.pi * 25))
