"""noiseacoustics: each indicator recomputed from its published definition, plus ISO 9613-2 Table 2."""

import math

import pytest

from morie.fn.noiseacoustics import (
    aircraft_dnl,
    atmospheric_absorption,
    attenuation_distance,
    barrier_attenuation,
    combine_levels,
    construction_noise,
    crtn_road_noise,
    day_evening_night,
    equivalent_level,
    foliage_attenuation,
    ground_attenuation,
    level_statistics,
    noise_annoyance,
    noise_health_risk,
    noise_map,
    noise_zones,
    point_source_level,
    sound_exposure_level,
    underwater_propagation,
    vibration_propagation,
)


def test_level_statistics_and_energy_sums():
    assert equivalent_level([60.0, 70.0], [3.0, 1.0]) == pytest.approx(10 * math.log10((3e6 + 1e7) / 4), abs=1e-12)
    s = level_statistics([50.0, 52.0, 55.0, 60.0, 71.0])
    # type-7 quantile: 0.9 * 4 = 3.6 -> 60 + 0.6 * 11
    assert (s.L10, s.L50, s.L90, s.Lmax, s.Lmin) == pytest.approx((66.6, 55.0, 50.8, 71.0, 50.0))
    assert sound_exposure_level([80.0] * 4, dt=0.5) == pytest.approx(80 + 10 * math.log10(2), abs=1e-12)
    assert combine_levels([60.0, 60.0, 60.0]) == pytest.approx(60 + 10 * math.log10(3), abs=1e-12)


def test_day_evening_night_windows():
    hourly = [45.0] * 7 + [65.0] * 12 + [60.0] * 4 + [45.0]
    r = day_evening_night(hourly)
    assert (r.Lday, r.Levening, r.Lnight) == pytest.approx((65.0, 60.0, 45.0), abs=1e-12)
    e = (12 * 10**6.5 + 4 * 10**6.5 + 8 * 10**5.5) / 24
    assert r.Lden == pytest.approx(10 * math.log10(e), abs=1e-12)
    # Ldn: day 07-22 holds 12 h at 65 and 3 h at 60; night 22-07 holds 1 h at 60 and 8 h at 45
    ldn = (12 * 10**6.5 + 3 * 10**6 + 10 * (10**6 + 8 * 10**4.5)) / 24
    assert r.Ldn == pytest.approx(10 * math.log10(ldn), abs=1e-12)
    cnel = (12 * 10**6.5 + 3 * 3 * 10**6 + 10 * (10**6 + 8 * 10**4.5)) / 24
    assert pytest.approx(10 * math.log10(cnel), abs=1e-12) == r.CNEL
    assert aircraft_dnl([90.0] * 10, [90.0]) == pytest.approx(10 * math.log10(2e10 / 86400), abs=1e-12)
    with pytest.raises(ValueError):
        day_evening_night([60.0] * 23)


def test_iso9613_absorption_table2():
    f = [1000 * 10 ** (0.3 * k) for k in range(-4, 4)]
    table = [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]  # ISO 9613-2 Table 2, 20 C, 70 %, dB/km
    got = [1000 * a for a in atmospheric_absorption(f, 20.0, 70.0)]
    assert got == pytest.approx(table, abs=0.05)


def test_propagation_terms():
    assert point_source_level(90.0, [50.0], alpha=0.005, dc=3.0) == pytest.approx(
        [90 + 3 - 20 * math.log10(50) - 11 - 0.25]
    )
    d = attenuation_distance(100.0, 40.0, alpha=0.01)
    assert point_source_level(100.0, [d], alpha=0.01)[0] == pytest.approx(40.0, abs=1e-9)
    g = ground_attenuation(50.0, 1.0, 3.0)
    dd = math.hypot(50, 2)
    assert g.A_gr == pytest.approx(4.8 - (4 / dd) * (17 + 300 / dd), abs=1e-12)
    assert ground_attenuation(5.0, 10.0, 10.0).A_gr == 0.0
    b = barrier_attenuation(20.0, 30.0, 49.0, 500.0, a=2.0)
    z = math.hypot(50, 2) - 49
    km = math.exp(-math.sqrt(20 * 30 * 49 / (2 * z)) / 2000)
    assert b.Dz == pytest.approx(10 * math.log10(3 + 20 / 0.68 * z * km), abs=1e-12)
    bd = barrier_attenuation(20.0, 30.0, 50.0, 1000.0, e=5.0)
    r5 = (5 * 0.34 / 5) ** 2
    zd = 5.0
    kmd = math.exp(-math.sqrt(20 * 30 * 50 / (2 * zd)) / 2000)
    assert bd.Dz == pytest.approx(min(25, 10 * math.log10(3 + 20 / 0.34 * (1 + r5) / (1 / 3 + r5) * zd * kmd)))
    assert barrier_attenuation(10.0, 10.0, 21.0, 1000.0).Dz == 0.0
    assert barrier_attenuation(100.0, 100.0, 150.0, 8000.0).Dz == 20.0
    assert [foliage_attenuation(x, 250) for x in (5.0, 15.0, 100.0, 500.0)] == pytest.approx([0.0, 1.0, 4.0, 8.0])
    with pytest.raises(ValueError):
        foliage_attenuation(50.0, 300)


def test_road_construction_vibration_underwater():
    r = crtn_road_noise(
        2000.0,
        speed=50.0,
        heavy_pct=10.0,
        gradient_pct=4.0,
        distance=20.0,
        receiver_height=4.0,
        soft_ground=0.5,
        angle=120.0,
        facade=True,
    )
    H = 2.25
    want = (
        42.2
        + 10 * math.log10(2000)
        + 33 * math.log10(50 + 40 + 10)
        + 10 * math.log10(2)
        - 68.8
        + 1.2
        - 10 * math.log10(math.hypot(23.5, 3.5) / 13.5)
        + 5.2 * 0.5 * math.log10((6 * H - 1.5) / 23.5)
        + 10 * math.log10(120 / 180)
        + 2.5
    )
    assert pytest.approx(want, abs=1e-12) == r.L10
    assert crtn_road_noise(20000.0, period="18h").corrections["basic"] == pytest.approx(29.1 + 43.0103, abs=1e-4)
    c = construction_noise(80.0, [25.0, 200.0], usage_factor=50.0)
    assert c.Lmax == pytest.approx([80 + 20 * math.log10(2), 80 - 20 * math.log10(4)])
    assert c.Leq[0] == pytest.approx(c.Lmax[0] + 10 * math.log10(0.5))
    assert vibration_propagation(90.0, [50.0]) == pytest.approx([90 - 30 * math.log10(2)])
    assert vibration_propagation(0.5, [25.0 * 4], kind="ppv") == pytest.approx([0.5 / 8])
    u = underwater_propagation(180.0, [100.0, 10000.0], 1.0, transition_range=1000.0)
    a = 0.11 / 2 + 44 / 4101 + 2.75e-4 + 0.003
    assert u.alpha == pytest.approx(a, abs=1e-15)
    assert pytest.approx([40 + a * 0.1, 60 + 10 + a * 10], abs=1e-12) == u.TL


def test_dose_response_map_zones():
    x = 70.0 - 42
    assert noise_annoyance([70.0], "aircraft") == pytest.approx([-9.199e-5 * x**3 + 3.932e-2 * x**2 + 0.2939 * x])
    assert noise_annoyance([65.0], "rail") == pytest.approx([7.239e-4 * 23**3 - 7.851e-3 * 23**2 + 0.1695 * 23])
    h = noise_health_risk([48.0, 58.0, 73.0], population=[50, 30, 20])
    assert h.rr == pytest.approx([1.0, 1.08**0.5, 1.08**2])
    xs = 0.3 * (1.08**0.5 - 1) + 0.2 * (1.08**2 - 1)
    assert h.paf == pytest.approx(xs / (1 + xs), abs=1e-15)
    m = noise_map([90.0, 95.0], [(0, 0), (30, 40)], [(0, 0), (30, 0)], alpha=0.002)
    e1 = 10 ** ((90 - 11 - 0.002) / 10) + 10 ** ((95 - 20 * math.log10(50) - 11 - 0.1) / 10)
    assert m[0] == pytest.approx(10 * math.log10(e1), abs=1e-12)
    z = noise_zones([54.9, 55.0, 74.9, 75.0])
    assert z.zone == [0, 1, 4, 5] and z.counts == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
