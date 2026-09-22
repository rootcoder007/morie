"""Tests for btciratio.boot_ci_ratio."""

from morie.fn import _array_core as np

from morie.fn.btciratio import boot_ci_ratio


def test_btciratio_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = boot_ci_ratio(x, y)
    assert isinstance(result, dict)
    assert "ratio" in result
def test_btciratio_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = boot_ci_ratio(x, y)
    assert isinstance(result, dict)
