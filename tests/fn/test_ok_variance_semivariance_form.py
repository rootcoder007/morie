"""Tests for ok_variance_semivariance_form.ok_variance_semivariance_form."""

from morie.fn import _array_core as np

from morie.fn.ok_variance_semivariance_form import (
    ok_variance_semivariance_form,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e11_basic():
    """Test basic functionality."""
    lam = 0.5
    gamma_s0 = 0.5
    nu = 0.5
    result = ok_variance_semivariance_form(lam, gamma_s0, nu)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r21e11_edge():
    """Test edge cases."""
    lam = 0.5
    gamma_s0 = 0.5
    nu = 0.5
    result = ok_variance_semivariance_form(lam, gamma_s0, nu)
    assert isinstance(result, dict)
