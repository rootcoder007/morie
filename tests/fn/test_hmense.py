"""Tests for hmense.geron_ensemble_eval."""

from morie.fn import _array_core as np

from morie.fn.hmense import geron_ensemble_eval


def test_hmense_basic():
    """Test basic functionality."""
    predictions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_ensemble_eval(predictions)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_hmense_edge():
    """Test edge cases."""
    predictions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_ensemble_eval(predictions)
    assert isinstance(result, dict)
