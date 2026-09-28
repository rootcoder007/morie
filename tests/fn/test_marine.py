"""marine: published formulas recomputed independently, and physical identities."""

import math

import pytest

from morie.fn._rng import random_normal
from morie.fn.marine import (
    bare_soil_index,
    co2_flux,
    depth_invariant_index,
    euphotic_depth,
    floating_algae_index,
    mangrove_vegetation_index,
    ocean_chlorophyll,
    oil_spill_drift,
    oxygen_solubility,
    plume_concentration,
    practical_salinity,
    sediment_transport,
    split_window_sst,
    vgpm_production,
)


def test_chlorophyll_and_production():
    x = math.log10(0.006 / 0.004)
    a = [0.2424, -2.7423, 1.8017, 0.0015, -1.2280]
    ref = 10 ** (a[0] + x * (a[1] + x * (a[2] + x * (a[3] + x * a[4]))))
    assert ocean_chlorophyll([[0.005, 0.006]], [0.004], "OC3M")[0] == pytest.approx(ref, rel=1e-13)
    # VGPM at 2 mg/m3 and 20 C by hand
    tot = 40.2 * 2.0**0.507
    z = 200 * tot**-0.293
    z = 568.2 * tot**-0.746 if z <= 102 else z
    T = 20.0
    pb = (
        1.2956
        + 0.2749 * T
        + 0.0617 * T**2
        - 0.0205 * T**3
        + 2.462e-3 * T**4
        - 1.348e-4 * T**5
        + 3.4132e-6 * T**6
        - 3.27e-8 * T**7
    )
    assert vgpm_production(2.0, 30.0, 20.0, 11.0) == pytest.approx(pb * 2 * 11 * 0.66125 * 30 / 34.1 * z, rel=1e-12)
    assert vgpm_production(1.0, 30.0, 29.0, 11.0) / vgpm_production(1.0, 30.0, 35.0, 11.0) == pytest.approx(1.0)
    assert euphotic_depth(0.2) == pytest.approx(math.log(100) / 0.2, rel=1e-15)


def test_gas_exchange():
    r = co2_flux(10.0, 25.0, 35.0, 380.0, 400.0)
    sc = 2116.8 - 136.25 * 25 + 4.7353 * 625 - 0.092307 * 15625 + 0.0007555 * 390625
    assert r.schmidt == pytest.approx(sc, rel=1e-14)
    tk = 2.9815
    k0 = math.exp(
        -58.0931 + 90.5069 / tk + 22.2940 * math.log(tk) + 35 * (0.027766 - 0.025888 * tk + 0.0050578 * tk**2)
    )
    assert r.flux == pytest.approx(0.24 * 0.251 * 100 * (sc / 660) ** -0.5 * k0 * -20, rel=1e-12)
    assert co2_flux(10.0, 25.0, 35.0, 400.0, 400.0).flux == 0.0
    t68 = 1.00024 * 8.0
    ts = math.log((298.15 - t68) / (273.15 + t68))
    A = [2.00907, 3.22014, 4.05010, 4.94457, -2.56847e-1, 3.88767]
    B = [-6.24523e-3, -7.37614e-3, -1.03410e-2, -8.17083e-3]
    lnc = sum(A[i] * ts**i for i in range(6)) + 33 * sum(B[i] * ts**i for i in range(4)) - 4.88682e-7 * 33**2
    assert oxygen_solubility(8.0, 33.0, "ml/l") == pytest.approx(math.exp(lnc), rel=1e-13)
    assert oxygen_solubility(25.0, 35.0) < oxygen_solubility(5.0, 35.0)


def test_practical_salinity_reference_point():
    # PSS-78 is defined so that C(35, 15 C IPTS-68, 0) = 42.914 mS/cm
    assert practical_salinity(42.914, 15.0 / 1.00024) == pytest.approx(35.0, abs=1e-6)
    assert practical_salinity([30.0, 40.0], 12.0) == sorted(practical_salinity([30.0, 40.0], 12.0))


def test_sst_sediment_and_indices():
    t11, t12, g, z = 288.0, 287.1, 15.0, 40.0
    c = [-255.0, 0.95, 0.07, 0.8]
    assert split_window_sst(t11, t12, c, g, z) == pytest.approx(
        c[0] + c[1] * t11 + c[2] * 0.9 * g + c[3] * 0.9 * (1 / math.cos(math.radians(z)) - 1), rel=1e-13
    )
    r = sediment_transport(2.0, 0.001)
    s = 2650 / 1025
    th = 2.0 / (1625 * 9.81 * 0.001)
    ds = 0.001 * ((s - 1) * 9.81 / 1.36e-6**2) ** (1 / 3)
    tc = 0.30 / (1 + 1.2 * ds) + 0.055 * (1 - math.exp(-0.02 * ds))
    assert r.theta == pytest.approx(th, rel=1e-14) and r.theta_cr == pytest.approx(tc, rel=1e-13)
    assert r.bedload == pytest.approx(8 * (th - tc) ** 1.5 * math.sqrt((s - 1) * 9.81 * 1e-9), rel=1e-12)
    assert sediment_transport(0.01, 0.001).bedload == 0.0
    assert floating_algae_index(0.05, 0.1, 0.03, (650.0, 850.0, 1250.0)) == pytest.approx(
        0.1 - (0.05 - 0.02 / 3), rel=1e-13
    )
    assert mangrove_vegetation_index(0.1, 0.4, 0.2) == pytest.approx(3.0, rel=1e-14)
    assert bare_soil_index(0.1, 0.2, 0.3, 0.4) == pytest.approx(0.2, rel=1e-14)


def test_depth_invariant_index_removes_depth():
    # sand at depths z with k_i = 0.08, k_j = 0.12: ln(L - Ls) = a - 2 k z (+ noise-free), so the index is constant
    zs = [2.0, 5.0, 9.0, 14.0]
    li = [0.01 + 0.3 * math.exp(-2 * 0.08 * z) for z in zs]
    lj = [0.005 + 0.2 * math.exp(-2 * 0.12 * z) for z in zs]
    r = depth_invariant_index(li, lj, 0.01, 0.005, li, lj)
    assert r.ratio == pytest.approx(0.08 / 0.12, rel=1e-12)
    assert max(r.dii) - min(r.dii) < 1e-12


def test_drift_and_plume():
    r = oil_spill_drift(0.0, 0.0, (0.1, -0.2), (5.0, 10.0), 6, 1.0, 3, diffusivity=4.0, seed=4)
    z = [float(v) for v in random_normal(36, seed=4)]
    sd = math.sqrt(2 * 4.0 * 3600)
    x0 = sum((0.25 * 3600 + sd * z[2 * (3 * s)]) for s in range(6))
    assert r.x[0] == pytest.approx(x0, rel=1e-12)
    # mass flux through a cross-section: h u integral of C dy = M exp(-k x / u)
    x, M, u, h, ky, k = 1500.0, 2.0, 0.25, 6.0, 1.5, 2e-5
    ys = [-1500 + 0.5 * i for i in range(6001)]
    c = plume_concentration([x] * len(ys), ys, M, u, h, ky, k)
    integral = 0.5 * (sum(c) - 0.5 * (c[0] + c[-1]))
    assert h * u * integral == pytest.approx(M * math.exp(-k * x / u), rel=1e-9)
    assert plume_concentration(-5.0, 0.0, M, u, h, ky) == 0.0
