"""Tests for pcfunc.pair_correlation_function."""

from morie.fn import _array_core as np

from morie.fn.pcfunc import pair_correlation_function


def test_pcfunc_basic():
    """Test basic functionality."""
    points = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    window = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    r = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = pair_correlation_function(points, window, r)
    assert isinstance(result, dict)
    assert "g" in result or "estimate" in result


def test_pcfunc_edge():
    """Test edge cases."""
    points = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    window = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    r = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = pair_correlation_function(points, window, r)
    assert isinstance(result, dict)
