"""Tests for ok_variance_covariance_form.ok_variance_covariance_form."""

from morie.fn import _array_core as np

from morie.fn.ok_variance_covariance_form import (
    ok_variance_covariance_form,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e8_basic():
    """Test basic functionality."""
    sigma2 = 0.5
    lam = 0.5
    cov_s0 = 0.5
    nu = 0.5
    result = ok_variance_covariance_form(sigma2, lam, cov_s0, nu)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e8_edge():
    """Test edge cases."""
    sigma2 = 0.5
    lam = 0.5
    cov_s0 = 0.5
    nu = 0.5
    result = ok_variance_covariance_form(sigma2, lam, cov_s0, nu)
    assert isinstance(result, dict)
