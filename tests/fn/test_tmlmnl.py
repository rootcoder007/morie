"""Tests for tmlmnl.tmle_machine_learning."""

from morie.fn import _array_core as np

from morie.fn.tmlmnl import tmle_machine_learning


def test_tmlmnl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tmle_machine_learning(y, D, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlmnl_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    D = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tmle_machine_learning(y, D, X)
    assert isinstance(result, dict)
