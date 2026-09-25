"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp15e2.mlpois."""

import math

import pytest

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp15e2 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_15_equation_2 as eq15_2,
    mlpois,
)

BETA0 = -1.5
BETA1 = 0.03
AGE = [25.0, 40.0, 55.0, 70.0, 33.0]
RACE = [0.0, 0.12, -0.08, 0.12, -0.08]
V = [0.05, -0.02, 0.01, 0.00, -0.04]
W = [-0.10, 0.20, 0.00, 0.15, -0.25]


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp15e2_basic():
    """log(lambda_i) = b0 + b1 age + beta(race) + v_i + W_i, eq. 15.2."""
    assert eq15_2 is mlpois
    res = mlpois(BETA0, BETA1, AGE, RACE, V, W)
    assert res["n"] == 5
    expect = [BETA0 + BETA1 * AGE[i] + RACE[i] + V[i] + W[i] for i in range(5)]
    for got, want in zip(res["lograte"], expect):
        assert abs(got - want) < 1e-12
    for got, want in zip(res["rate"], expect):
        assert abs(got - math.exp(want)) < 1e-12


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp15e2_edge():
    """The age slope is exactly b1 per year; ragged inputs are rejected."""
    zeros = [0.0] * 5
    res = mlpois(0.0, 0.0, AGE, zeros, zeros, zeros)
    assert res["lograte"] == zeros
    assert res["rate"] == [1.0] * 5
    # a one-year rise in age multiplies the rate by exp(b1)
    up = mlpois(BETA0, BETA1, [a + 1.0 for a in AGE], RACE, V, W)
    base = mlpois(BETA0, BETA1, AGE, RACE, V, W)
    for i in range(5):
        assert abs(up["lograte"][i] - base["lograte"][i] - BETA1) < 1e-12
        assert abs(up["rate"][i] / base["rate"][i] - math.exp(BETA1)) < 1e-12
    with pytest.raises(ValueError):
        mlpois(BETA0, BETA1, AGE, RACE, V, W[:4])
