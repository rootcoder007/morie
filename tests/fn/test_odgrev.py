"""Tests for odgrev.outbreak_detection."""

from morie.fn import _array_core as np

from morie.fn.odgrev import outbreak_detection


def test_odgrev_basic():
    """Test basic functionality."""
    counts = np.array([1, 0, 2, 2, 3, 3, 1, 0, 4, 3, 2, 3, 0, 1, 1, 3, 3, 0, 1, 1, 3, 2, 4, 0, 4, 3, 1, 3, 0, 2, 2, 4, 1, 4, 2, 2, 3, 0, 0, 3])
    result = outbreak_detection(counts)
    assert isinstance(result, dict)
    assert "cp_prob" in result


def test_odgrev_edge():
    """Test edge cases."""
    counts = np.array([1, 0, 2, 2, 3, 3, 1, 0, 4, 3, 2, 3, 0, 1, 1, 3, 3, 0, 1, 1, 3, 2, 4, 0, 4, 3, 1, 3, 0, 2, 2, 4, 1, 4, 2, 2, 3, 0, 0, 3])
    result = outbreak_detection(counts)
    assert isinstance(result, dict)
