"""Tests for rgfd2.rangayyan_second_diff."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_second_diff


def test_rgfd2_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_second_diff(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgfd2_edge():
    """Test edge cases."""
    result = rangayyan_second_diff(np.array([42.0]))
    assert result["n"] == 1
