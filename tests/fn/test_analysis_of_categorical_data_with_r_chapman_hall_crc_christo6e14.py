"""Tests for analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e14.analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_14."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e14 import (
    analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_14,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e14_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # For one item pair, all three inputs are scalars (log(mu_ab) = b0 + bW_a + bY_b)
    b0 = float(rng.normal(0, 1))
    beta_w_a = float(rng.normal(0, 1))
    beta_y_b = float(rng.normal(0, 1))
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_14(
        b0, beta_w_a, beta_y_b
    )
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    # exp(...) is always positive and finite for finite inputs
    assert math.isfinite(val)
    assert val > 0
    assert "method" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e14_edge():
    """Test edge cases."""
    # All-zero parameters: exp(0 + 0 + 0) = 1
    b0 = 0.0
    beta_w_a = 0.0
    beta_y_b = 0.0
    result = analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_14(
        b0, beta_w_a, beta_y_b
    )
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    assert math.isfinite(val)
    assert val > 0
    assert "method" in result
