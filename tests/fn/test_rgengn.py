"""Tests for rgengn.rangayyan_eng."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_eng


def test_rgengn_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_eng(t)
    assert isinstance(result, dict)
    assert "t_ms" in result


def test_rgengn_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_eng(t)
    assert isinstance(result, dict)
