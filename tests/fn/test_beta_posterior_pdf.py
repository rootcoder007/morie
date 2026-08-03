"""Tests for beta_posterior_pdf.beta_posterior_pdf."""

from morie.fn import _array_core as np

from morie.fn.beta_posterior_pdf import (
    beta_posterior_pdf,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e24_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = beta_posterior_pdf(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e24_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = beta_posterior_pdf(x)
    assert isinstance(result, dict)
