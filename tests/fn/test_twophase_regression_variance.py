"""Tests for twophase_regression_variance (Brus 2022, eq. 11.7)."""

import pytest

from morie.fn.twophase_regression_variance import twophase_regression_variance


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e7_basic():
    """V = (1 - n1/N) S2(z)/n1 + (1 - n2/n1) S2(e)/n2."""
    v = twophase_regression_variance(4.0, 200, 1.5, 50, 10000)
    assert v["value"] == pytest.approx((1 - 0.02) * 4.0 / 200 + (1 - 0.25) * 1.5 / 50, rel=1e-15)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e7_edge():
    """A census second phase (n2 = n1) leaves only the first-phase term;
    n2 > n1 and negative variances raise."""
    assert twophase_regression_variance(4.0, 100, 9.0, 100, 1000)["value"] == pytest.approx(0.9 * 0.04, rel=1e-15)
    with pytest.raises(ValueError):
        twophase_regression_variance(1.0, 10, 1.0, 20, 100)
    with pytest.raises(ValueError):
        twophase_regression_variance(-1.0, 10, 1.0, 5, 100)
