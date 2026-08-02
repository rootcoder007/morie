"""Tests for quntf.quantile_function."""

from morie.fn import _array_core as np

from morie.fn.quntf import quantile_function


def test_quntf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = quantile_function(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_quntf_edge():
    """Test edge cases."""
    result = quantile_function(np.array([42.0]))
    assert result["n"] == 1
