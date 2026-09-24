"""Tests for interaction_logit.interaction_logit."""

import math

from morie.fn import _array_core as np

from morie.fn.interaction_logit import (
    interaction_logit,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e22_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # 9 coefficients: b0, b1, b2, b3, b4, b5, b6, b7, b8
    b = rng.normal(0, 1, 9).tolist()
    # x1, x2 are scalar categorical indicators for a single observation;
    # z1, z2 are scalar categorical indicators for the second variable.
    x1 = 1
    x2 = 0
    z1 = 1
    z2 = 0
    result = interaction_logit(b, x1, x2, z1, z2)
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    assert isinstance(val, (int, float))
    assert math.isfinite(val)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo2e22_edge():
    """Test edge cases."""
    # Edge case: all coefficients zero => logit == 0 regardless of categories.
    b = [0.0] * 9
    x1 = 0
    x2 = 0
    z1 = 0
    z2 = 0
    result = interaction_logit(b, x1, x2, z1, z2)
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    assert isinstance(val, (int, float))
    assert math.isclose(val, 0.0)
