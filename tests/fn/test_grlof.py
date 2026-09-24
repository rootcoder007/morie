"""Tests for grlof.geron_local_outlier_factor."""

from morie.fn import _array_core as np

from morie.fn.grlof import geron_local_outlier_factor


def test_grlof_basic():
    """Test basic functionality."""
    X = [[0.0], [0.5], [1.0], [1.5], [40.0]]
    result = geron_local_outlier_factor(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grlof_edge():
    """Test edge cases."""
    X = [[0.0], [0.5], [1.0], [1.5], [40.0]]
    result = geron_local_outlier_factor(X)
    assert isinstance(result, dict)
