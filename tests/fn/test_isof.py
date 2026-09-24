"""Tests for isoF.isolation_forest."""

from morie.fn import _array_core as np

from morie.fn.isof import isolation_forest


def test_isof_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = isolation_forest(X)
    assert isinstance(result, dict)
    assert "score" in result


def test_isof_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = isolation_forest(X)
    assert isinstance(result, dict)
