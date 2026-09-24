"""Tests for hmocsv.geron_one_class_svm."""

from morie.fn import _array_core as np

from morie.fn.hmocsv import geron_one_class_svm


def test_hmocsv_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_one_class_svm(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "alpha" in result


def test_hmocsv_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_one_class_svm(X)
    assert isinstance(result, dict)
