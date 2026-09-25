"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp17e1.logitre."""

import math

import pytest

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp17e1 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_17_equation_1 as eq17_1,
    logitre,
)

G0, G1, G2 = -0.4, 0.8, -0.15
D = [0.0, 0.5, 1.0, 1.5, 2.0]
X = [1.2, -0.3, 0.0, 2.4, 0.9]
R = [0.05, -0.10, 0.00, 0.20, -0.02]


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp17e1_basic():
    """logit(p_i) = g0 + g1 d_i + g2 x_i + R_i, eq. 17.1."""
    assert eq17_1 is logitre
    res = logitre(G0, G1, D, G2, X, R)
    assert res["n"] == 5
    expect = [G0 + G1 * D[i] + G2 * X[i] + R[i] for i in range(5)]
    for got, want in zip(res["eta"], expect):
        assert abs(got - want) < 1e-12
    for got, want in zip(res["p"], expect):
        assert abs(got - 1.0 / (1.0 + math.exp(-want))) < 1e-12
        assert 0.0 < got < 1.0


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp17e1_edge():
    """A zero linear predictor gives p = 1/2; ragged inputs are rejected."""
    zeros = [0.0] * 3
    res = logitre(0.0, 1.0, zeros, 1.0, zeros, zeros)
    assert res["eta"] == zeros
    assert res["p"] == [0.5, 0.5, 0.5]
    # the odds ratio for a unit rise in d is exactly exp(g1)
    base = logitre(G0, G1, D, G2, X, R)
    up = logitre(G0, G1, [d + 1.0 for d in D], G2, X, R)
    for i in range(5):
        ob = base["p"][i] / (1.0 - base["p"][i])
        ou = up["p"][i] / (1.0 - up["p"][i])
        assert abs(ou / ob - math.exp(G1)) < 1e-9
    with pytest.raises(ValueError):
        logitre(G0, G1, D, G2, X[:4], R)
