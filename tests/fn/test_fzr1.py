"""Tests for fzr1.fauzi_r1_integral."""

from morie.fn import _array_core as np

from morie.fn.fzr1 import fauzi_r1_integral


def test_fzr1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_r1_integral(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_fzr1_edge():
    """Test edge cases."""
    result = fauzi_r1_integral(np.array([42.0]))
    assert result["n"] == 1
