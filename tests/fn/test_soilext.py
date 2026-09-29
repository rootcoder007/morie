"""soilext: formulas and limiting behaviour."""

import math

import pytest

from morie.fn.soilext import cesium_redistribution, ec25_correction, ipcc_soil_carbon, rweq_wind_erosion


def test_ec_and_cesium():
    assert ec25_correction(2.0, 25.0) == pytest.approx(2.0 * (0.447 + 1.4034 * math.exp(-25 / 26.815)), rel=1e-14)
    assert ec25_correction(2.0, 25.0, "linear") == 2.0
    assert ec25_correction([1.0, 2.0], 10.0, "linear") == pytest.approx([1 / 0.7, 2 / 0.7], rel=1e-14)
    assert cesium_redistribution(2400.0, 2400.0, 0.25, 1300.0, 50.0) == 0.0
    x = 100 * (3000 - 2100) / 3000
    assert cesium_redistribution(2100.0, 3000.0, 0.25, 1400.0, 55.0, p=1.2) == pytest.approx(
        -10 * 0.25 * 1400 * x / (100 * 55 * 1.2), rel=1e-14
    )
    assert cesium_redistribution(2100.0, 3000.0, 0, 0, 55.0, model="profile", h=40.0) == pytest.approx(
        10 / 55 * 40 * math.log(0.7), rel=1e-14
    )


def test_ipcc_and_rweq():
    r = ipcc_soil_carbon(70.0, 0.8, 1.15, 1.04, area=5.0, soc_initial=300.0)
    s = 70 * 0.8 * 1.15 * 1.04 * 5
    assert r.stock == pytest.approx(s, rel=1e-14) and r.annual_change == pytest.approx((s - 300) / 20, rel=1e-13)
    w = rweq_wind_erosion(15.0, None, None, 0.7, 0.8, field_length=1e6, sand=60, silt=25, clay=15, om=2, caco3=1)
    ef = (29.09 + 0.31 * 60 + 0.17 * 25 + 0.33 * 4 - 2.59 * 2 - 0.95) / 100
    assert w.ef == pytest.approx(ef, rel=1e-14)
    assert w.transport == pytest.approx(w.q_max, rel=1e-12)  # a very long field reaches maximum transport
    assert w.soil_loss == pytest.approx(0.0, abs=1e-12)
