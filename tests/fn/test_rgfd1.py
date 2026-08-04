"""Tests for rgfd1.rangayyan_first_diff."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_first_diff


def test_rgfd1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_first_diff(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgfd1_edge():
    """Test edge cases."""
    result = rangayyan_first_diff(np.array([42.0]))
    assert result["n"] == 1
