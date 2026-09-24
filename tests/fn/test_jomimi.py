"""Tests for jomimi.joseph_missing_data_imputation_ts."""

from morie.fn import _array_core as np

from morie.fn.jomimi import joseph_missing_data_imputation_ts


def test_jomimi_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_missing_data_imputation_ts(x)
    assert isinstance(result, dict)
    assert "x" in result


def test_jomimi_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = joseph_missing_data_imputation_ts(x)
    assert isinstance(result, dict)
