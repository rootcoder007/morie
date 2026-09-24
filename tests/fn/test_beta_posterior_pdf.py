"""Tests for beta_posterior_pdf.beta_posterior_pdf."""

import math

from morie.fn.beta_posterior_pdf import (
    beta_posterior_pdf,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e24_basic():
    """Test basic functionality."""
    n = 100
    z = 30
    c = 1.0
    d = 1.0
    p = 0.3
    result = beta_posterior_pdf(p, z, n, c, d)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0


def test_the_r_series_dick_j_brus_spatial_sampling_with_r12e24_edge():
    """Test edge cases."""
    n = 50
    z = 10
    c = 2.0
    d = 2.0
    p = 0.5
    result = beta_posterior_pdf(p, z, n, c, d)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0
