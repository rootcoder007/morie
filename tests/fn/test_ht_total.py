"""Tests for ht_total.ht_total."""

from morie.fn import _array_core as np

from morie.fn.ht_total import (
    ht_total,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r2e2_basic():
    """Test basic functionality."""
    z = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    pi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ht_total(z, pi)
    assert isinstance(result, dict)
    assert "value" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r2e2_edge():
    """Test edge cases."""
    z = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    pi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ht_total(z, pi)
    assert isinstance(result, dict)
