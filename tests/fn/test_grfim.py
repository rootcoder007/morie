"""Tests for grfim.geron_feature_importance_mdi."""

from morie.fn import _array_core as np

from morie.fn.grfim import geron_feature_importance_mdi


def test_grfim_basic():
    """Test basic functionality."""
    tree_importances = [[10.0, 0.0], [0.0, 1.0]]
    result = geron_feature_importance_mdi(tree_importances)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grfim_edge():
    """Test edge cases."""
    tree_importances = [[10.0, 0.0], [0.0, 1.0]]
    result = geron_feature_importance_mdi(tree_importances)
    assert isinstance(result, dict)
