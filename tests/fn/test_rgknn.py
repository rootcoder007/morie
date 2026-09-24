"""Tests for rgknn.rangayyan_knn_classifier."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_knn_classifier


def test_rgknn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 3
    X_train = rng.normal(0, 1, (n, p))
    y_train = np.random.default_rng(43).integers(0, 2, n)
    X_test = rng.normal(0, 1, p)
    k = 5
    result = rangayyan_knn_classifier(X_train, y_train, X_test, k)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rgknn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 20, 3
    X_train = rng.normal(0, 1, (n, p))
    y_train = np.random.default_rng(43).integers(0, 3, n)
    X_test = rng.normal(0, 1, p)
    k = 3
    result = rangayyan_knn_classifier(X_train, y_train, X_test, k)
    assert isinstance(result, dict)
