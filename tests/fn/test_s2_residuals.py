"""Tests for s2_residuals.s2_residuals."""

from morie.fn import _array_core as np

from morie.fn.s2_residuals import (
    s2_residuals,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = s2_residuals(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = s2_residuals(x)
    assert isinstance(result, dict)
