"""climidx: normalisation and least-squares identities."""

import math

import pytest

from morie.fn._rng import random_normal
from morie.fn.climidx import co2_curve_fit, nao_station_index


def test_nao_is_difference_of_z_scores():
    z = [float(v) for v in random_normal(48, seed=4)]
    s = [1015 + 4 * v for v in z[:24]]
    n = [1003 - 6 * v for v in z[24:]]
    months = [i % 12 + 1 for i in range(24)]
    idx = nao_station_index(s, n, months)
    for m in (1, 7):
        rows = [i for i in range(24) if months[i] == m]
        ms, mn = sum(s[i] for i in rows) / 2, sum(n[i] for i in rows) / 2
        ss = math.sqrt(sum((s[i] - ms) ** 2 for i in rows))
        sn = math.sqrt(sum((n[i] - mn) ** 2 for i in rows))
        for i in rows:
            assert idx[i] == pytest.approx((s[i] - ms) / ss - (n[i] - mn) / sn, rel=1e-12)


def test_co2_fit_recovers_exact_curve():
    t = [1990 + i / 12 for i in range(120)]
    y = [
        355 + 1.6 * (v - 1995) + 0.01 * (v - 1995) ** 2 + 3 * math.sin(2 * math.pi * v) - math.cos(4 * math.pi * v)
        for v in t
    ]
    r = co2_curve_fit(t, y, 3, 2)
    assert max(abs(e) for e in r.residuals) < 1e-9
    for v, g in zip(t, r.growth_rate):
        assert g == pytest.approx(1.6 + 0.02 * (v - 1995), abs=1e-9)
    assert sum(r.seasonal) / len(r.seasonal) == pytest.approx(0.0, abs=1e-9)
