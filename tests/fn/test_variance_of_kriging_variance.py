"""Tests for variance_of_kriging_variance (Brus 2022, eq. 24.3)."""

import pytest

from morie.fn.variance_of_kriging_variance import variance_of_kriging_variance


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e3_basic():
    """VKV = sum_ij Cov(th_i, th_j) dV/dth_i dV/dth_j, the delta-method
    variance of the kriging variance, summed here term by term."""
    C = [[0.04, 0.01, -0.005], [0.01, 0.09, 0.002], [-0.005, 0.002, 0.25]]
    g = [1.5, -0.4, 0.2]
    ref = sum(C[i][j] * g[i] * g[j] for i in range(3) for j in range(3))
    assert variance_of_kriging_variance(C, g)["value"] == pytest.approx(ref, rel=1e-14)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e3_edge():
    """One parameter: Var(theta) (dV/dtheta)^2; mismatched shapes raise."""
    assert variance_of_kriging_variance([[0.5]], [2.0])["value"] == pytest.approx(2.0, rel=1e-15)
    with pytest.raises(ValueError):
        variance_of_kriging_variance([[1.0, 0.0], [0.0, 1.0]], [1.0])
