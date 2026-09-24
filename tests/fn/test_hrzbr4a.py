"""Tests for hrzbr4a.horowitz_binary_response_model."""

from morie.fn import _array_core as np

from morie.fn.hrzbr4a import horowitz_binary_response_model


def test_hrzbr4a_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    result = horowitz_binary_response_model(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hrzbr4a_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 20, 2
    X = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    result = horowitz_binary_response_model(X, y)
    assert isinstance(result, dict)
