"""Tests for sschin.chained_imputation."""

from morie.fn import _array_core as np

from morie.fn.sschin import chained_imputation


def test_sschin_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = chained_imputation(time, event, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_sschin_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.random.default_rng(42).normal(0.0, 1.0, 40)
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = chained_imputation(time, event, X)
    assert isinstance(result, dict)
