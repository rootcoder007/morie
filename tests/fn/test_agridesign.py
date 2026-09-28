"""Tests for agridesign: agricultural and landscape design formulas."""

import math

from morie.fn.agridesign import (
    buffer_strip_width,
    drain_water_table,
    hooghoudt_spacing,
    range_condition,
    rotation_score,
    variable_rate_zones,
    wetland_area_pkc,
)


def test_buffer_width_inverts_first_order_trapping():
    r = buffer_strip_width(0.75, k=0.08)
    assert abs(1 - math.exp(-0.08 * r.width) - 0.75) <= 1e-12
    w, e = [3, 6, 10, 15, 30], [0.3, 0.45, 0.62, 0.71, 0.93]
    y = [-math.log(1 - v) for v in e]
    k = sum(a * b for a, b in zip(w, y)) / sum(a * a for a in w)
    assert abs(buffer_strip_width(0.8, widths=w, efficiencies=e).k - k) <= 1e-15


def test_wetland_plug_flow_and_tanks_in_series():
    r = wetland_area_pkc(850, 120, 15, 25, 3)
    q = 365 * 850 / r.area
    assert abs((15 - 3) / (120 - 3) - math.exp(-25 / q)) <= 1e-12
    t = wetland_area_pkc(850, 120, 15, 25, 3, n_tanks=3)
    q3 = 365 * 850 / t.area
    assert abs((15 - 3) / (120 - 3) - (1 + 25 / (3 * q3)) ** -3) <= 1e-12
    assert t.area > r.area


def test_range_condition_classes():
    assert range_condition([10, 25, 5, 60], [30, 20, 40, 10]).score == 10 + 20 + 5 + 10
    assert range_condition([10, 25, 5, 60], [30, 20, 40, 10]).condition == "fair"
    assert range_condition([100], [100]).condition == "excellent"


def test_rotation_score_and_screens():
    E = [[0, 1, 2, -1], [0.5, -2, 1, 0], [1, 1, 0, 2], [0, 0.2, 0.3, -1]]
    seq = [0, 2, 1, 0, 3]
    r = rotation_score(seq, E, min_return=[3, 1, 1, 1], forbidden=[(2, 1)])
    assert r.score == E[0][2] + E[2][1] + E[1][0] + E[0][3] + E[3][0]
    assert ("sequence", 1) in r.violations and ("return", 0) in r.violations
    assert rotation_score(seq, E, min_return=[2, 1, 1, 1]).violations == []
    assert rotation_score([0, 1], E, max_frequency=[0.4, 1, 1, 1]).feasible is False


def test_variable_rate_zones_stanford():
    y = [5.1, 7.3, 6.2, 9.9, 8.4, 4.4, 6.6, 7.7, 10.2]
    sn = [20, 30, 25, 40, 22, 10, 35, 28, 50]
    r = variable_rate_zones(y, 3, soil_n=sn, efficiency=0.6)
    for z in range(3):
        idx = [i for i in range(9) if r.zone[i] == z]
        my = sum(y[i] for i in idx) / len(idx)
        ms = sum(sn[i] for i in idx) / len(idx)
        assert abs(r.rate[z] - max((20 * my - ms) / 0.6, 0)) <= 1e-12
    assert sorted(r.zone) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_hooghoudt_satisfies_its_equation():
    r = hooghoudt_spacing(0.005, 1.1, 0.3, 0.8, 1.2)
    d = r.equivalent_depth
    assert abs(0.005 * r.spacing**2 - (8 * 0.8 * d * 1.1 + 4 * 0.3 * 1.1**2)) <= 1e-9
    D, L = 1.2, r.spacing
    assert abs(d - D / (1 + D / L * (8 / math.pi * math.log(D / 0.1) - 3.4))) <= 1e-12
    h = drain_water_table(0.005, r.spacing, 0.3, 0.8, d)
    assert abs(h - 1.1) <= 1e-9
    deep = hooghoudt_spacing(0.004, 0.6, 0.6, 1.4, 50.0)
    assert 50.0 / deep.spacing > 0.3
    assert abs(deep.equivalent_depth - math.pi * deep.spacing / (8 * (math.log(deep.spacing / 0.1) - 1.15))) <= 1e-12
