"""Tests for eslboo.esl_bootstrap_err."""

from morie.fn import _array_core as np

from morie.fn.eslboo import esl_bootstrap_err


def test_eslboo_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_bootstrap_err(X, y)
    assert isinstance(result, dict)
    assert "err_boot" in result
def test_eslboo_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_bootstrap_err(X, y)
    assert isinstance(result, dict)
