"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e3.epiar."""

import math

import pytest

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e3 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_3 as eq18_3,
    epiar,
)

B0, B1 = 0.3, 0.75
I_LAG = [1.0, 4.0, 10.0, 25.0, 100.0]
B1I = [0.02, -0.05, 0.00, 0.10, -0.20]


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e3_basic():
    """log(f) = b0 + b1 log(I_lag) + b1i, eq. 18.3."""
    assert eq18_3 is epiar
    res = epiar(B0, B1, I_LAG, B1I)
    assert res["n"] == 5
    expect = [B0 + B1 * math.log(I_LAG[i]) + B1I[i] for i in range(5)]
    for got, want in zip(res["logf"], expect):
        assert abs(got - want) < 1e-12
    for got, want in zip(res["f"], expect):
        assert abs(got - math.exp(want)) < 1e-12
    # log(1) = 0, so the first area is the intercept plus its own effect
    assert abs(res["logf"][0] - (B0 + B1I[0])) < 1e-12


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e3_edge():
    """b1 is the elasticity of f in the lagged count; non-positive lags fail."""
    base = epiar(B0, B1, I_LAG, B1I)
    dbl = epiar(B0, B1, [2.0 * v for v in I_LAG], B1I)
    for i in range(5):
        assert abs(dbl["logf"][i] - base["logf"][i]
                   - B1 * math.log(2.0)) < 1e-12
        assert abs(dbl["f"][i] / base["f"][i] - 2.0 ** B1) < 1e-9
    flat = epiar(0.0, 0.0, I_LAG, [0.0] * 5)
    assert flat["logf"] == [0.0] * 5
    assert flat["f"] == [1.0] * 5
    with pytest.raises(ValueError):
        epiar(B0, B1, [1.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        epiar(B0, B1, I_LAG, B1I[:4])
