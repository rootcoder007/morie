"""Tests for s2_residuals.s2_residuals."""

from morie.fn import _array_core as np

from morie.fn.s2_residuals import (
    s2_residuals,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e8_basic():
    """Test basic functionality."""
    e = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    result = s2_residuals(e, n)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r11e8_edge():
    """Test edge cases."""
    e = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    n = 5
    result = s2_residuals(e, n)
    assert isinstance(result, dict)
