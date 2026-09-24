"""Tests for category_prob_from_cumulative.category_prob_from_cumulative."""

import math

from morie.fn import _array_core as np

from morie.fn.category_prob_from_cumulative import (
    category_prob_from_cumulative,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e12_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    cum_probs = np.array([0.1, 0.3, 0.6, 0.8, 1.0])
    j = 2
    result = category_prob_from_cumulative(cum_probs, j)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e12_edge():
    """Test edge cases."""
    cum_probs = np.array([0.2, 0.5, 1.0])
    j = 1
    result = category_prob_from_cumulative(cum_probs, j)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert 0.0 <= result["value"] <= 1.0
