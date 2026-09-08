"""Tests for nested_anova_prediction.nested_anova_prediction."""

from morie.fn import _array_core as np

from morie.fn.nested_anova_prediction import (
    nested_anova_prediction,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = nested_anova_prediction(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = nested_anova_prediction(x)
    assert isinstance(result, dict)
