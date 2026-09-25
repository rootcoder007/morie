"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp11e1.facrisk."""

import math

import pytest

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp11e1 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_11_equation_1 as eq11_1,
    facrisk,
)

ALPHA0 = -0.25
# five areas, two latent factors: W is n by L, phi is one value per factor
W = [[1.0, 0.0],
     [0.5, 0.5],
     [0.0, 1.0],
     [0.25, 0.75],
     [0.8, 0.2]]
PHI = [0.4, -0.6]


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp11e1_basic():
    """log(theta_i) = alpha_0 + sum_l w_il phi_l, eq. 11.1."""
    assert eq11_1 is facrisk
    res = facrisk(ALPHA0, W, PHI)
    assert res["n"] == 5
    assert res["n_components"] == 2
    expect = [ALPHA0 + sum(row[l] * PHI[l] for l in range(2)) for row in W]
    for got, want in zip(res["logrisk"], expect):
        assert abs(got - want) < 1e-12
    for got, want in zip(res["risk"], expect):
        assert abs(got - math.exp(want)) < 1e-12
    # an all-zero weight row leaves the intercept alone
    flat = facrisk(ALPHA0, [[0.0, 0.0]], PHI)
    assert abs(flat["logrisk"][0] - ALPHA0) < 1e-12
    assert abs(flat["risk"][0] - math.exp(ALPHA0)) < 1e-12


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp11e1_edge():
    """Zero factors give exp(alpha_0) everywhere; ragged W is rejected."""
    res = facrisk(0.0, [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
    assert res["logrisk"] == [0.0, 0.0]
    assert res["risk"] == [1.0, 1.0]
    assert res["n_components"] == 2
    # the log-risk is linear in phi
    a = facrisk(ALPHA0, W, [0.4, -0.6])["logrisk"]
    b = facrisk(ALPHA0, W, [0.8, -1.2])["logrisk"]
    for i in range(5):
        assert abs((b[i] - ALPHA0) - 2.0 * (a[i] - ALPHA0)) < 1e-12
    with pytest.raises(ValueError):
        facrisk(ALPHA0, [[1.0, 0.0], [1.0]], PHI)
