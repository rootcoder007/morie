"""Tests for morie.fn.clusterdetect (elliptic, flexible, normal scans; Cuzick-Edwards; Lawson-Waller; GAM)."""

import math

import pytest

from morie.fn.clusterdetect import (
    cuzick_edwards,
    elliptic_scan,
    fixed_circle_scan,
    flex_zones,
    flexscan,
    lawson_waller,
    normal_scan,
)


def _llr(yin, ein, ty):
    return yin * math.log(yin / ein) + (ty - yin) * math.log((ty - yin) / (ty - ein))


def test_elliptic_scan_penalised_llr():
    P = [(float(i), 0.0) for i in range(6)] + [(float(i), 1.0) for i in range(6)]
    y = [9, 8, 9, 8, 0, 0, 0, 1, 0, 0, 1, 0]
    r = elliptic_scan(P, y, [10] * 12, nsim=0)
    c = r.clusters[0]
    assert sorted(c["zone"]) == [0, 1, 2, 3] and (c["shape"], c["angle"]) == (2.0, 180.0)
    ty = sum(y)
    assert c["llr"] == pytest.approx(_llr(34, 4 * ty / 12, ty) * math.sqrt(8 / 9), abs=1e-12)


def test_flexscan_and_zones():
    P = [(float(i), 0.0) for i in range(5)]
    W = [[1 if abs(i - j) == 1 else 0 for j in range(5)] for i in range(5)]
    Z = flex_zones(P, W, k=3)
    assert [0, 2] not in Z and [0, 1, 2] in Z
    r = flexscan(P, [8, 7, 1, 1, 1], [10] * 5, W, k=3, nsim=0)
    assert r.clusters[0]["zone"] == [0, 1]
    assert r.clusters[0]["llr"] == pytest.approx(_llr(15, 7.2, 18), abs=1e-12)


def test_normal_scan_hand_value():
    x = [5.0, 6.0, 1.0, 2.0, 1.5, 0.5]
    r = normal_scan([(float(i), 0.0) for i in range(6)], x, nsim=0)
    s0 = sum(v * v for v in x) / 6 - (sum(x) / 6) ** 2
    sz = (0.5 + 1.25) / 6
    assert r.zone == [0, 1]
    assert r.llr == pytest.approx(3 * math.log(s0 / sz), abs=1e-12)


def test_cuzick_edwards_and_lawson_waller():
    P = [(0, 0), (1, 0), (10, 0), (11, 0), (20, 0), (21, 0)]
    r = cuzick_edwards(P, [1, 1, 0, 0, 1, 0], k=1, nsim=19)
    assert (r.statistic, r.expected) == (2, 1.2)
    assert 0 < r.pvalue <= 1
    lw = lawson_waller([4, 2, 1, 1], [2, 2, 2, 2], [1.0, 0.5, 0.25, 0.25])
    assert (lw.U, lw.variance) == (1.5, 0.75)
    assert lw.z == pytest.approx(1.5 / math.sqrt(0.75), abs=1e-15)
    assert lawson_waller([4, 2, 1, 1], [2, 2, 2, 2], [1.0, 0.5, 0.25, 0.25], conditional=False).variance == 2.75


def test_fixed_circle_gam_poisson_tail():
    P = [(0.0, 0.0), (0.5, 0.0), (5.0, 5.0), (9.0, 1.0)]
    r = fixed_circle_scan(P, [6, 5, 1, 0], [10, 10, 10, 10], [1.0], overlap=0.5, alpha=1.0)
    for c in r.circles:
        O, E = c["observed"], c["expected"]
        tail = 1 - sum(math.exp(-E) * E**j / math.factorial(j) for j in range(int(O)))
        assert c["pvalue"] == pytest.approx(tail, abs=1e-12)
    with pytest.raises(ValueError):
        fixed_circle_scan(P, [1] * 4, [1] * 4, [1.0], method="bad")
