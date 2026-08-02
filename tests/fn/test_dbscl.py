"""Tests for dbscl.dbscan_clustering."""

from morie.fn import _array_core as np

from morie.fn.dbscl import dbscan_clustering


def test_dbscl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = dbscan_clustering(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_dbscl_edge():
    """Test edge cases."""
    result = dbscan_clustering(np.array([42.0]))
    assert result["n"] == 1
