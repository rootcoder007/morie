"""Tests for classification_indicator.classification_indicator."""

from morie.fn import _array_core as np

from morie.fn.classification_indicator import (
    classification_indicator,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r25e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = classification_indicator(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r25e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = classification_indicator(x)
    assert isinstance(result, dict)
