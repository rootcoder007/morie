"""Tests for survipa.ipa_brier."""

from morie.fn import _array_core as np

from morie.fn.survipa import ipa_brier


def test_survipa_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    predicted_S = np.random.default_rng(42).normal(0.0, 1.0, 40)
    eval_time = 0.1
    result = ipa_brier(time, event, predicted_S, eval_time)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_survipa_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    predicted_S = np.random.default_rng(42).normal(0.0, 1.0, 40)
    eval_time = 0.1
    result = ipa_brier(time, event, predicted_S, eval_time)
    assert isinstance(result, dict)
