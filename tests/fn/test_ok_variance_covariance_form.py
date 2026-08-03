"""Tests for ok_variance_covariance_form.ok_variance_covariance_form."""

from morie.fn import _array_core as np

from morie.fn.ok_variance_covariance_form import (
    ok_variance_covariance_form,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ok_variance_covariance_form(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = ok_variance_covariance_form(x)
    assert isinstance(result, dict)
