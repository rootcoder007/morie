"""Tests for bowbn.bow_ban_theorem."""

from morie.fn import _array_core as np

from morie.fn.bowbn import bow_ban_theorem


def test_bowbn_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bow_ban_theorem(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_bowbn_edge():
    """Test edge cases."""
    result = bow_ban_theorem(np.array([42.0]))
    assert result["n"] == 1
