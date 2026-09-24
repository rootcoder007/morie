"""Tests for product_multinomial_pmf.product_multinomial_pmf."""

import math

from morie.fn import _array_core as np
from morie.fn.product_multinomial_pmf import product_multinomial_pmf


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Count table: I rows, J columns of nonnegative integer counts
    I, J = 3, 4
    count_table = rng.integers(0, 10, size=(I, J))
    # Conditional probability table: shape (I, J), each row sums to 1
    cond_prob_table = rng.uniform(0, 1, size=(I, J))
    cond_prob_table = [
        [v / sum(row) for v in row] for row in cond_prob_table
    ]
    result = product_multinomial_pmf(count_table, cond_prob_table)
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    assert math.isfinite(val)
    assert 0.0 <= val <= 1.0


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e3_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    # Smallest nontrivial case: a single row (I=1) with J=2 categories
    I, J = 1, 2
    count_table = rng.integers(0, 5, size=(I, J))
    cond_prob_table = rng.uniform(0, 1, size=(I, J))
    cond_prob_table = [
        [v / sum(row) for v in row] for row in cond_prob_table
    ]
    result = product_multinomial_pmf(count_table, cond_prob_table)
    assert isinstance(result, dict)
    assert "value" in result
    val = result["value"]
    assert math.isfinite(val)
    assert 0.0 <= val <= 1.0
