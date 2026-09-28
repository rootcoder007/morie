"""Tests for morie.fn.crimegeo (CrimeStat journey to crime, circle hypothesis, hit score, risk terrain)."""

import math

import pytest

from morie.fn.crimegeo import (
    circle_hypothesis,
    jtc_calibrate,
    jtc_decay,
    jtc_surface,
    risk_layers,
    risk_terrain,
    search_cost,
)


def test_decay_functions_match_levine_equations():
    d = 3.0
    assert jtc_decay(d, "linear") == [1.9 - 0.06 * d]
    assert jtc_decay(d, "negative_exponential")[0] == pytest.approx(1.89 * math.exp(-0.06 * d), abs=1e-15)
    z = (d - 4.2) / 4.6
    assert jtc_decay(d, "normal")[0] == pytest.approx(29.5 / (4.6 * math.sqrt(2 * math.pi)) * math.exp(-z * z / 2))
    ln = 8.6 / (d * d * 4.6 * math.sqrt(2 * math.pi)) * math.exp(-((math.log(d * d) - 4.2) ** 2) / (2 * 4.6**2))
    assert jtc_decay(d, "lognormal")[0] == pytest.approx(ln, abs=1e-15)
    assert jtc_decay([0.2, 3.0], "truncated_negative_exponential") == pytest.approx(
        [13.8 / 0.4 * 0.2, 13.8 * math.exp(-0.2 * 2.6)], abs=1e-12
    )
    assert jtc_decay(0.0, "lognormal") == [0.0]
    with pytest.raises(ValueError):
        jtc_decay(1.0, "gravity")


def test_surface_and_hit_score():
    r = jtc_surface([(0, 0), (2, 0)], [(1, 0), (5, 0)], "linear", {"A": 1.0, "B": -0.1})
    assert r.score == pytest.approx([1.8, 1.2]) and r.peak == (1.0, 0.0)
    m = jtc_surface([(0, 0)], [(1, 1)], "linear", {"A": 1.0, "B": -0.1}, metric="manhattan")
    assert m.score == pytest.approx([0.8])
    s = search_cost([5.0, 3.0, 9.0, 3.0], 1)
    assert (s.hit_percent, s.hit_percent_ties, s.rank, s.ties) == (50.0, 100.0, 3, 2)


def test_calibration_grouped_regressions():
    D = [0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 4.5, 5.5]
    r = jtc_calibrate(D, [0, 1, 2, 3, 4, 5, 6])
    mid = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    pct = [10.0, 20.0, 30.0, 20.0, 10.0, 10.0]
    assert r.pct == pct
    mx, my = sum(mid) / 6, sum(pct) / 6
    b = sum((a - mx) * (c - my) for a, c in zip(mid, pct)) / sum((a - mx) ** 2 for a in mid)
    assert r.params["linear"]["B"] == pytest.approx(b, abs=1e-12)
    t = r.params["truncated_negative_exponential"]
    x, lp = mid[3:], [math.log(v) for v in pct[3:]]
    mx, my = sum(x) / 3, sum(lp) / 3
    C = -sum((a - mx) * (c - my) for a, c in zip(x, lp)) / sum((a - mx) ** 2 for a in x)
    assert (t["cutoff"], t["peak"]) == (2.5, 30.0) and t["C"] == pytest.approx(C, abs=1e-12)
    assert r.best in r.rss


def test_circle_hypothesis():
    r = circle_hypothesis([(0, 0), (4, 0), (2, 1)], home=(2, -1))
    assert (r.center, r.radius, r.marauder, r.share_inside) == ((2.0, 0.0), 2.0, True, 1.0)
    tri = circle_hypothesis([(0, 0), (1, 0), (0.5, math.sqrt(3) / 2)], home=(5, 5))
    assert tri.marauder is False and tri.share_inside == pytest.approx(2 / 3)


def test_risk_terrain_saturated_design():
    assert risk_layers([(0, 0), (5, 0)], [(1, 0), (1.5, 0)], radius=2) == [1, 0]
    r = risk_terrain([1, 2, 4, 8], [[0, 1, 0, 1], [0, 0, 1, 1]])
    assert r.rrv == pytest.approx([2.0, 4.0], abs=1e-9)
    assert r.risk_score == pytest.approx([1.0, 2.0, 4.0, 8.0], abs=1e-8)
    assert r.deviance == pytest.approx(0.0, abs=1e-9)
