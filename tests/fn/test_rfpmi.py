"""Tests for rfpmi.rf_permutation_importance."""

from morie.fn import _array_core as np

from morie.fn.rfpmi import rf_permutation_importance


def test_rfpmi_basic():
    """Test basic functionality."""
    forest = 5
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rf_permutation_importance(forest, X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rfpmi_edge():
    """Test edge cases."""
    forest = 5
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rf_permutation_importance(forest, X, y)
    assert isinstance(result, dict)
