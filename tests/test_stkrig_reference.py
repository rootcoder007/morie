"""Space-time covariance models, empirical space-time variogram and space-time kriging.

Checked against gstat (vgmST separable / productSum / metric / sumMetric,
krigeST ordinary and simple, variogramST) to 1e-14;
tests/cross/test-morie_vs_gstat_st.R repeats that in R.
"""

import math

import pytest

from morie.fn.stkrig import st_covariance, st_kriging_cv, st_variogram
from morie.fn.zsstk import st_kriging
from morie.fn.zsstv import st_kriging_var

SEP = {
    "type": "separable",
    "sill": 2.0,
    "space": {"model": "Exp", "range": 1.0},
    "time": {"model": "Gau", "range": 2.0},
}


def test_covariance_models():
    assert st_covariance(1.0, 1.0, SEP) == pytest.approx(2.0 * math.exp(-1.0) * math.exp(-0.25), abs=1e-15)
    ps = {
        "type": "productSum",
        "k": 0.5,
        "space": {"model": "Exp", "range": 1.0},
        "time": {"model": "Exp", "range": 1.0},
    }
    cs, ct = math.exp(-2.0), math.exp(-1.0)
    assert st_covariance(2.0, 1.0, ps) == pytest.approx(ct + cs + 0.5 * ct * cs, abs=1e-15)
    met = {"type": "metric", "stAni": 2.0, "joint": {"model": "Exp", "range": 1.0, "nugget": 0.3}}
    assert st_covariance(3.0, 2.0, met) == pytest.approx(math.exp(-5.0), abs=1e-15)
    assert st_covariance(0.0, 0.0, met) == pytest.approx(1.3, abs=1e-15)


def test_kriging_interpolates_and_documented():
    m = {
        "type": "separable",
        "sill": 1.0,
        "space": {"model": "Exp", "range": 1.0},
        "time": {"model": "Exp", "range": 2.0},
    }
    P, T, z = [(0, 0), (1, 0), (0, 1)], [0, 0, 1], [1.0, 2.0, 1.5]
    r = st_kriging(z, P, T, P, T, m)
    assert r.local_values == pytest.approx(z, abs=1e-12)
    assert max(abs(v) for v in r.extra["variance"]) < 1e-12
    v = st_kriging_var(z, P, T, [(0.5, 0.5)], [0.5], m).local_values[0]
    assert v == pytest.approx(st_kriging(z, P, T, [(0.5, 0.5)], [0.5], m).extra["variance"][0], abs=1e-15)


def test_variogram_zero_bin_and_counts():
    z = [[1.0, 2.0, 4.0], [2.0, 2.5, 3.0]]
    v = st_variogram(z, [(0, 0), (1, 0), (2, 0)], [0, 1], tlags=[0, 1], boundaries=[0, 1.5, 3])
    assert v.np == [0, 4, 2, 3, 4, 2]
    # lag 1, zero distance: half mean of (1-2)^2, (2-2.5)^2, (4-3)^2
    assert v.gamma[3] == pytest.approx(0.5 * (1 + 0.25 + 1) / 3, abs=1e-15)


def test_cross_validation_residuals():
    m = {"type": "metric", "stAni": 1.0, "joint": {"model": "Exp", "psill": 1.0, "range": 1.0}}
    z = [1.0, 2.0, 1.5, 1.2]
    cv = st_kriging_cv(z, [(0, 0), (1, 0), (0, 1), (1, 1)], [0, 0, 1, 1], m)
    assert cv.residuals == pytest.approx([a - b for a, b in zip(z, cv.prediction)], abs=1e-15)
    assert cv.rmse == pytest.approx(math.sqrt(sum(r * r for r in cv.residuals) / 4), abs=1e-15)
