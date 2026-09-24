"""Tests for nested_anova_prediction.nested_anova_prediction."""

import math

from morie.fn import _array_core as np

from morie.fn.nested_anova_prediction import (
    nested_anova_prediction,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    mu = 0.0
    a_i = rng.normal(0, 1)
    b_ij = rng.normal(0, 1)
    c_ijk = rng.normal(0, 1)
    eps = rng.normal(0, 1)
    result = nested_anova_prediction(mu, a_i, b_ij, c_ijk, eps)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_the_r_series_dick_j_brus_spatial_sampling_with_r24e1_edge():
    """Test edge cases."""
    result = nested_anova_prediction(0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] == 0.0
