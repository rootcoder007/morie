"""Tests for wsmstn.wasserman_sufficient."""

from morie.fn import _array_core as np

from morie.fn.wsmstn import wasserman_sufficient


def test_wsmstn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_sufficient(x)
    assert isinstance(result, dict)
    assert "T1" in result or "T1" in result


def test_wsmstn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_sufficient(x)
    assert isinstance(result, dict)
