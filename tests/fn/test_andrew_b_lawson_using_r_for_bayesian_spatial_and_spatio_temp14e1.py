"""Tests for andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14e1.mvfacmu."""

import math

import pytest

from morie.fn.andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14e1 import (
    andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_equation_1 as eq14_1,
    mvfacmu,
)

# four areas by three diseases of expected counts
E = [[10.0, 20.0, 5.0],
     [8.0, 16.0, 4.0],
     [12.0, 24.0, 6.0],
     [30.0, 15.0, 9.0]]
LAM = [1.0, 0.5, -0.25]
F = [0.2, -0.4, 0.0, 0.6]


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14e1_basic():
    """mu_ik = e_ik exp(lambda_k f_i), eq. 14.1."""
    assert eq14_1 is mvfacmu
    res = mvfacmu(E, LAM, F)
    assert res["n"] == 4
    assert res["n_disease"] == 3
    for i in range(4):
        for k in range(3):
            want = math.exp(LAM[k] * F[i])
            assert abs(res["rho"][i][k] - want) < 1e-12
            assert abs(res["mu"][i][k] - E[i][k] * want) < 1e-12
    # a zero factor leaves the expected counts untouched in that area
    assert res["rho"][2] == [1.0, 1.0, 1.0]
    assert res["mu"][2] == E[2]


def test_andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14e1_edge():
    """Zero loadings collapse rho to one; mismatched shapes are rejected."""
    res = mvfacmu(E, [0.0, 0.0, 0.0], F)
    assert res["rho"] == [[1.0] * 3 for _ in range(4)]
    assert res["mu"] == E
    # one disease, one area
    one = mvfacmu([[7.0]], [2.0], [0.5])
    assert one["n"] == 1 and one["n_disease"] == 1
    assert abs(one["rho"][0][0] - math.exp(1.0)) < 1e-12
    assert abs(one["mu"][0][0] - 7.0 * math.exp(1.0)) < 1e-12
    with pytest.raises(ValueError):
        mvfacmu(E, LAM, [0.2, -0.4])
    with pytest.raises(ValueError):
        mvfacmu(E, [1.0, 0.5], F)
