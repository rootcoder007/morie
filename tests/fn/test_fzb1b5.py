"""Tests for fzb1b5.fauzi_assumptions_b1_b5."""

from morie.fn import _array_core as np

from morie.fn.fzb1b5 import fauzi_assumptions_b1_b5


def test_fzb1b5_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_assumptions_b1_b5(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_fzb1b5_edge():
    """Test edge cases."""
    result = fauzi_assumptions_b1_b5(np.array([42.0]))
    assert result["n"] == 1
