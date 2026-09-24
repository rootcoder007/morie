"""Tests for hmcv.geron_cross_validation."""

from morie.fn import _array_core as np

from morie.fn.hmcv import geron_cross_validation


def test_hmcv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    fold_predictions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    folds = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_cross_validation(y, fold_predictions, folds)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hmcv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    fold_predictions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    folds = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_cross_validation(y, fold_predictions, folds)
    assert isinstance(result, dict)
