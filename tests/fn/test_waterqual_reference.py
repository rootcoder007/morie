"""waterqual: indices recomputed from their published definitions; DO saturation against Benson-Krause values."""

import math

import pytest

from morie.fn.waterqual import (
    carlson_tsi,
    ccme_wqi,
    constituent_load,
    do_saturation,
    irrigation_water_quality,
    removal_efficiency,
    suspended_solids,
    total_dissolved_solids,
    weighted_arithmetic_wqi,
)


def test_ccme_wqi():
    nan = float("nan")
    X = [[1.0, 8.0, 0.2], [3.0, 5.0, nan], [1.5, 9.0, 0.9]]
    r = ccme_wqi(X, [2.0, 6.0, 0.5], ["max", "min", "max"])
    assert pytest.approx(100.0) == r.F1
    assert pytest.approx(100 * 3 / 8) == r.F2
    nse = (0.5 + 0.2 + 0.8) / 8
    assert r.nse == pytest.approx(nse, abs=1e-15)
    F3 = nse / (0.01 * nse + 0.01)
    assert r.wqi == pytest.approx(100 - math.sqrt(1e4 + 37.5**2 + F3**2) / 1.732, abs=1e-12)
    ok = ccme_wqi([[1.0], [1.5]], [2.0])
    assert (ok.wqi, ok.category) == (100.0, "Excellent")


def test_weighted_wqi_and_do():
    r = weighted_arithmetic_wqi([7.5, 250.0, 2.0], [8.5, 500.0, 5.0], ideal=[7.0, 0.0, 0.0])
    K = 1 / (1 / 8.5 + 1 / 500 + 1 / 5)
    assert sum(r.w) == pytest.approx(1.0, abs=1e-15)
    assert r.wqi == pytest.approx(K * (100 / 3 / 8.5 + 50 / 500 + 40 / 5), abs=1e-12)
    # Benson and Krause (1984) / APHA 4500-O values at 1 atm, fresh water
    assert [round(v, 2) for v in do_saturation([5.0, 10.0, 15.0, 25.0]).cs] == [12.77, 11.29, 10.08, 8.26]
    T = 293.15
    sea = do_saturation([20.0], salinity=35.0).cs[0]
    assert math.log(sea) == pytest.approx(
        math.log(do_saturation([20.0]).cs[0]) - 35 * (0.017674 - 10.754 / T + 2140.7 / T**2), abs=1e-14
    )
    assert round(sea, 3) == 7.396
    lp = do_saturation([20.0], pressure_atm=0.8).cs[0]
    pwv = math.exp(11.8571 - 3840.70 / T - 216961 / T**2)
    th = 0.000975 - 1.426e-5 * 20 + 6.436e-8 * 400
    base = do_saturation([20.0]).cs[0]
    assert lp == pytest.approx(base * 0.8 * (1 - pwv / 0.8) * (1 - th * 0.8) / ((1 - pwv) * (1 - th)), abs=1e-12)
    assert do_saturation([20.0], measured=[4.546]).percent_saturation[0] == pytest.approx(100 * 4.546 / base)


def test_carlson_irrigation_solids_loads():
    c = carlson_tsi(chla_ugl=100.0)
    assert c.tsi_chl == pytest.approx(9.81 * math.log(100) + 30.6) and c.state == "hypereutrophic"
    assert carlson_tsi(secchi_m=8.0).state == "oligotrophic"
    with pytest.raises(ValueError):
        carlson_tsi()
    i = irrigation_water_quality(na=6.0, ca=3.0, mg=5.0, k=0.5, hco3=4.0, co3=1.0, ec_us_cm=200.0)
    assert (i.sar, i.rsc, i.kelly_ratio, i.magnesium_hazard) == pytest.approx((3.0, -3.0, 0.75, 62.5))
    assert i.percent_na == pytest.approx(650 / 14.5) and i.salinity_class == "C1"
    t = total_dissolved_solids(ec_us_cm=1000.0, k=0.7, ions=[12.0, 3.0], bicarbonate=61.0)
    assert (t.tds_ec, t.tds_ions) == pytest.approx((700.0, 15 + 0.4917 * 61))
    s = suspended_solids(1523.4, 1510.2, 250.0, ignited_mg=1514.6)
    assert (s.tss, s.vss, s.fss) == pytest.approx((52.8, 35.2, 17.6))
    L = constituent_load([2.0, 4.0, 1.0], [1.0, 3.0, 2.0], dt=10.0)
    assert (L.load, L.fwmc) == pytest.approx((160.0, 16 / 6))
    r = removal_efficiency([50.0], [5.0])
    assert (r.percent, r.lrv) == ([90.0], [1.0])
