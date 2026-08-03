"""Tests for product_multinomial_pmf.product_multinomial_pmf."""

from morie.fn import _array_core as np

from morie.fn.product_multinomial_pmf import (
    product_multinomial_pmf,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e3_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = product_multinomial_pmf(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo3e3_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = product_multinomial_pmf(x)
    assert isinstance(result, dict)
