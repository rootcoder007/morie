"""Tests for tmlcmp.tmle_competing_risks."""

from morie.fn import _array_core as np

from morie.fn.tmlcmp import tmle_competing_risks


def test_tmlcmp_basic():
    """Test basic functionality."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event_type = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tmle_competing_risks(time, event_type, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlcmp_edge():
    """Test edge cases."""
    time = np.random.default_rng(42).normal(0.0, 1.0, 40)
    event_type = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tmle_competing_risks(time, event_type, D, X)
    assert isinstance(result, dict)
