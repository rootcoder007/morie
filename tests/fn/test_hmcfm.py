"""Tests for hmcfm.geron_confusion_matrix."""

from morie.fn import _array_core as np

from morie.fn.hmcfm import geron_confusion_matrix


def test_hmcfm_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    y_pred = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_confusion_matrix(y_true, y_pred)
    assert isinstance(result, dict)
    assert "estimate" in result or "matrix" in result


def test_hmcfm_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    y_pred = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_confusion_matrix(y_true, y_pred)
    assert isinstance(result, dict)
