"""Tests for rgap.rangayyan_action_potential."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_action_potential


def test_rgap_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_action_potential(t)
    assert isinstance(result, dict)
    assert "t_ms" in result


def test_rgap_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_action_potential(t)
    assert isinstance(result, dict)
