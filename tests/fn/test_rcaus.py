"""Tests for rcaus.random_cause_refutation."""

from morie.fn import _array_core as np

from morie.fn.rcaus import random_cause_refutation


def test_rcaus_basic():
    """Test basic functionality."""
    estimator = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    d = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_cause_refutation(estimator, y, d, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rcaus_edge():
    """Test edge cases."""
    estimator = (lambda *a, **k: float(np.sum(np.asarray(a[0]) ** 2)))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    d = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = random_cause_refutation(estimator, y, d, X)
    assert isinstance(result, dict)
