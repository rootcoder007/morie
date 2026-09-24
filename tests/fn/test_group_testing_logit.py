"""Tests for group_testing_logit.group_testing_logit."""

import math

from morie.fn import _array_core as np

from morie.fn.group_testing_logit import (
    group_testing_logit,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e32_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    xs = rng.normal(0, 1, 3)
    bs = [0.5, -0.3, 0.2]
    b0 = 0.1
    result = group_testing_logit(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e32_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    xs = rng.normal(0, 1, 2)
    bs = [1.0, -1.0]
    b0 = 0.0
    result = group_testing_logit(b0, bs, xs)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
