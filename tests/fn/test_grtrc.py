"""Tests for grtrc.geron_tree_classification_leaf."""

from morie.fn import _array_core as np

from morie.fn.grtrc import geron_tree_classification_leaf


def test_grtrc_basic():
    """Test basic functionality."""
    y = [1] * 49 + [2] * 5
    result = geron_tree_classification_leaf(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grtrc_edge():
    """Test edge cases."""
    y = [1] * 49 + [2] * 5
    result = geron_tree_classification_leaf(y)
    assert isinstance(result, dict)
