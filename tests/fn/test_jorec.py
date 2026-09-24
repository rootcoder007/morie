"""Tests for jorec.joseph_recursive_multistep."""

from morie.fn import _array_core as np

from morie.fn.jorec import joseph_recursive_multistep


def test_jorec_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    x = rng.normal(0, 1, 100)
    result = joseph_recursive_multistep(x, lags=[1, 2, 3, 5], horizon=3)
    assert isinstance(result, dict)
    assert "forecast" in result or "statistic" in result or "estimate" in result


def test_jorec_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    x = rng.normal(0, 1, 50)
    result = joseph_recursive_multistep(x, lags=[1, 2], horizon=2)
    assert isinstance(result, dict)
    assert "forecast" in result or "statistic" in result or "estimate" in result
