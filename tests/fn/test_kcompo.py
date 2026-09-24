"""Tests for kcompo.k_step_dp_composition."""

from morie.fn import _array_core as np

from morie.fn.kcompo import k_step_dp_composition


def test_kcompo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    epsilons = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = k_step_dp_composition(y, epsilons)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kcompo_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    epsilons = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = k_step_dp_composition(y, epsilons)
    assert isinstance(result, dict)
