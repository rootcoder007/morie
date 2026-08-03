"""Tests for beta_pdf.beta_pdf."""

from morie.fn import _array_core as np

from morie.fn.beta_pdf import (
    beta_pdf,
)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = beta_pdf(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo1e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = beta_pdf(x)
    assert isinstance(result, dict)
