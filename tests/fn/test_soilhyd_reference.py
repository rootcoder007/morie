"""soilhyd: every function recomputed from its published equation, plus the defining identities."""

import math

import pytest

from morie.fn.soilhyd import (
    infiltration,
    rusle,
    saxton_rawls,
    scs_runoff,
    sobel_filter,
    soil_carbon,
    soil_chemistry,
    usda_texture,
    van_genuchten,
)


def test_saxton_rawls_table1():
    S, C, OM = 0.40, 0.20, 2.5
    r = saxton_rawls(S, C, OM)
    t15t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    assert r.theta1500 == pytest.approx(t15t + 0.14 * t15t - 0.02, abs=1e-15)
    assert r.theta33 == pytest.approx(t33t + 1.283 * t33t**2 - 0.374 * t33t - 0.015, abs=1e-15)
    assert r.rho_n == pytest.approx((1 - r.theta_s) * 2.65, abs=1e-15)
    assert pytest.approx((math.log(1500) - math.log(33)) / (math.log(r.theta33) - math.log(r.theta1500))) == r.B
    assert r.ksat == pytest.approx(1930 * (r.theta_s - r.theta33) ** (3 - 1 / r.B))
    # tension 33 kPa at theta33 through psi = A theta^-B
    assert r.A * r.theta33 ** (-r.B) == pytest.approx(33.0, rel=1e-12)
    assert saxton_rawls(S, C, OM, density_factor=1.1).theta_s_df < r.theta_s


def test_van_genuchten_and_infiltration():
    vg = van_genuchten([0.0, 50.0, 1e12], 0.05, 0.45, 0.02, 1.5, ks=10.0)
    assert vg.theta[0] == 0.45 and vg.theta[2] == pytest.approx(0.05, abs=1e-3)
    se = (1 + (0.02 * 50) ** 1.5) ** (-1 / 3)
    assert vg.theta[1] == pytest.approx(0.05 + 0.4 * se, abs=1e-15)
    assert vg.K[1] == pytest.approx(10 * se**0.5 * (1 - (1 - se**3) ** (1 / 3)) ** 2, abs=1e-15)
    h = infiltration([0.0, 1.0], model="horton", f0=10.0, fc=2.0, k=1.0)
    assert h.rate[0] == 10.0 and h.cumulative[1] == pytest.approx(2 + 8 * (1 - math.exp(-1)))
    p = infiltration([4.0], model="philip", S=2.0, A=0.5)
    assert (p.cumulative[0], p.rate[0]) == pytest.approx((6.0, 1.0))
    g = infiltration([2.0], model="green_ampt", Ks=1.0, psi=10.0, dtheta=0.3)
    F = g.cumulative[0]
    assert F - 3.0 * math.log(1 + F / 3.0) == pytest.approx(2.0, abs=1e-12)
    assert g.rate[0] == pytest.approx(1 + 3.0 / F)


def test_runoff_erosion_chemistry_carbon():
    q = scs_runoff([10.0, 50.0], 80)
    S = 25400 / 80 - 254
    assert q.runoff[0] == 0.0 and q.runoff[1] == pytest.approx((50 - 0.2 * S) ** 2 / (50 + 0.8 * S), abs=1e-12)
    e = rusle(100.0, 0.3, 0.2, 0.5, slope_length=50.0, slope_pct=4.0)
    th = math.atan(0.04)
    LS = (50 / 22.13) ** 0.4 * (65.41 * math.sin(th) ** 2 + 4.56 * math.sin(th) + 0.065)
    assert (e.m, e.LS, e.A) == pytest.approx((0.4, LS, 100 * 0.3 * LS * 0.2 * 0.5))
    c = soil_chemistry(na=10.0, ca=4.0, mg=4.0, k=1.0, h_al=1.0, na_ex=2.0)
    assert (c.sar, c.cec, c.esp) == pytest.approx((5.0, 20.0, 10.0))
    s = soil_carbon(
        2.0,
        1.2,
        20.0,
        coarse_fraction=0.1,
        clay_pct=20.0,
        silt_pct=30.0,
        aggregate_fractions=[0.2, 0.5, 0.3],
        aggregate_diameters=[3.0, 1.0, 0.25],
    )
    assert s.stock == pytest.approx(43.2) and s.pieri_si == pytest.approx(6.896)
    assert s.mwd == pytest.approx(0.6 + 0.5 + 0.075)


def test_texture_and_sobel():
    cases = {
        (92, 5, 3): "sand",
        (82, 12, 6): "loamy sand",
        (65, 25, 10): "sandy loam",
        (40, 40, 20): "loam",
        (20, 65, 15): "silt loam",
        (5, 88, 7): "silt",
        (60, 15, 25): "sandy clay loam",
        (35, 35, 30): "clay loam",
        (10, 60, 30): "silty clay loam",
        (50, 10, 40): "sandy clay",
        (5, 50, 45): "silty clay",
        (20, 20, 60): "clay",
    }
    for (s, si, c), want in cases.items():
        assert usda_texture(s, si, c) == want
    with pytest.raises(ValueError):
        usda_texture(50, 30, 30)
    f = sobel_filter([[0, 0, 0], [1, 1, 1], [2, 2, 2]], res=2.0)
    assert (f.gx[1][1], f.gy[1][1], f.magnitude[1][1]) == pytest.approx((0.0, 0.5, 0.5))
    assert math.isnan(f.magnitude[0][0])
