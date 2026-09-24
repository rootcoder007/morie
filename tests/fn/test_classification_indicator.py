"""Tests for classification_indicator.classification_indicator."""

import math

from morie.fn import _array_core as np

from morie.fn.classification_indicator import (
    classification_indicator,
)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r25e8_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    c_hat = list(rng.integers(0, 3, n))
    c_true = list(rng.integers(0, 3, n))
    u = 1
    result = classification_indicator(c_hat, c_true, u)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0


def test_the_r_series_dick_j_brus_spatial_sampling_with_r25e8_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 100
    c_hat = list(rng.integers(0, 3, n))
    c_true = list(rng.integers(0, 3, n))
    u = 0
    result = classification_indicator(c_hat, c_true, u)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0
