"""Tests for grstk.geron_stacking_predictor."""

from morie.fn import _array_core as np

from morie.fn.grstk import geron_stacking_predictor


def test_grstk_basic():
    """Test basic functionality."""
    base_preds = [[1.5, 0.5], [2.5, 1.5], [3.5, 2.5]]
    y = [1.0, 2.0, 3.0]
    result = geron_stacking_predictor(base_preds, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grstk_edge():
    """Test edge cases."""
    base_preds = [[1.5, 0.5], [2.5, 1.5], [3.5, 2.5]]
    y = [1.0, 2.0, 3.0]
    result = geron_stacking_predictor(base_preds, y)
    assert isinstance(result, dict)
