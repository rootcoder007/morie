"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4.epiarnb."""

import math

import pytest

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_18_equation_4 as eq18_4,
    epiarnb,
)

B0, B1 = 0.3, 0.75
I_LAG = [1.0, 4.0, 10.0, 25.0, 100.0]
NB_LAG = [3.0, 0.0, 15.0, 5.0, 50.0]
B1I = [0.02, -0.05, 0.00, 0.10, -0.20]


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4_basic():
    """log(f) = b0 + b1 log(I_lag + neighbour lag) + b1i, eq. 18.4."""
    assert eq18_4 is epiarnb
    res = epiarnb(B0, B1, I_LAG, NB_LAG, B1I)
    assert res["n"] == 5
    assert res["total_lag"] == [4.0, 4.0, 25.0, 30.0, 150.0]
    expect = [B0 + B1 * math.log(I_LAG[i] + NB_LAG[i]) + B1I[i]
              for i in range(5)]
    for got, want in zip(res["logf"], expect):
        assert abs(got - want) < 1e-12
    for got, want in zip(res["f"], expect):
        assert abs(got - math.exp(want)) < 1e-12


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e4_edge():
    """With no neighbours eq. 18.4 reduces to eq. 18.3; bad lags fail."""
    from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp18e3 \
        import epiar
    plain = epiar(B0, B1, I_LAG, B1I)
    none = epiarnb(B0, B1, I_LAG, [0.0] * 5, B1I)
    for i in range(5):
        assert abs(none["logf"][i] - plain["logf"][i]) < 1e-12
    assert none["total_lag"] == I_LAG
    # only the total matters: moving count from own to neighbour is a no-op
    moved = epiarnb(B0, B1, [v - 0.5 for v in I_LAG],
                    [n + 0.5 for n in NB_LAG], B1I)
    base = epiarnb(B0, B1, I_LAG, NB_LAG, B1I)
    for i in range(5):
        assert abs(moved["logf"][i] - base["logf"][i]) < 1e-12
    with pytest.raises(ValueError):
        epiarnb(B0, B1, [1.0, -1.0], [0.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        epiarnb(B0, B1, I_LAG, NB_LAG, B1I[:4])
