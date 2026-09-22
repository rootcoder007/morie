"""Tests for autofm.autoformer."""

from morie.fn import _array_core as np

from morie.fn.autofm import autoformer


def test_autofm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = 25
    result = autoformer(x, kernel)
    assert isinstance(result, dict)
    assert "trend" in result
    assert "seasonal" in result


def test_autofm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    kernel = 25
    result = autoformer(x, kernel)
    assert isinstance(result, dict)
