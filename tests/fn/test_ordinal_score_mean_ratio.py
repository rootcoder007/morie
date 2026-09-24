"""Tests for ordinal_score_mean_ratio.ordinal_score_mean_ratio."""

import math

from morie.fn import _array_core as np

from morie.fn.ordinal_score_mean_ratio import (
    ordinal_score_mean_ratio,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e12_basic():
    """Test basic functionality."""
    beta_z_j = 0.5
    beta_z_jp = 0.1
    beta_xz_i = 0.2
    s_j = 1.0
    s_jp = 2.0
    result = ordinal_score_mean_ratio(beta_z_j, beta_z_jp, beta_xz_i, s_j, s_jp)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] > 0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo4e12_edge():
    """Test edge cases."""
    # All-zero inputs yield exp(0) = 1.0
    result = ordinal_score_mean_ratio(0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert math.isclose(result["value"], 1.0)
