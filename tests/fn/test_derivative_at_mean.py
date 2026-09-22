"""Tests for derivative_at_mean.derivative_at_mean."""

from morie.fn import _array_core as np

from morie.fn.derivative_at_mean import derivative_at_mean


def test_ca4e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    ybar = rng.uniform(0.1, 0.9, size=100).mean()
    b = rng.normal(0, 1, size=100).mean()
    result = derivative_at_mean(ybar, b)
    expected = ybar * (1 - ybar) * b
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], float)
    assert abs(result["value"] - expected) < 1e-12


def test_ca4e9_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    ybar = 0.4
    b = 2.0
    result = derivative_at_mean(ybar, b)
    expected = ybar * (1 - ybar) * b
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], float)
    assert abs(result["value"] - expected) < 1e-12
