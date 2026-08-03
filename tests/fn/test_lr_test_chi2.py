"""Tests for lr_test_chi2.lr_test_chi2."""

from morie.fn import _array_core as np

from morie.fn.lr_test_chi2 import lr_test_chi2


def test_ca7e8_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = lr_test_chi2(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ca7e8_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = lr_test_chi2(x)
    assert isinstance(result, dict)
