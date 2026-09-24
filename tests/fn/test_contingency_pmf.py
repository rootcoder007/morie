"""Tests for contingency_pmf.contingency_pmf."""

import math

from morie.fn import _array_core as np

from morie.fn.contingency_pmf import (
    contingency_pmf,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Build a 2x2 probability table (rows sum to 1, all nonnegative)
    prob_table = np.array([[0.2, 0.3],
                           [0.1, 0.4]])
    # Build a 2x2 count table of nonnegative integers summing to some total
    count_table = np.array([[5, 7],
                           [3, 10]])
    result = contingency_pmf(count_table, prob_table)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))
    assert 0.0 <= float(result["value"]) <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # 1x2 cell table (degenerate but still valid: one row, two columns)
    prob_table = np.array([[0.6, 0.4]])
    count_table = np.array([[3, 2]])
    result = contingency_pmf(count_table, prob_table)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))
    assert 0.0 <= float(result["value"]) <= 1.0
