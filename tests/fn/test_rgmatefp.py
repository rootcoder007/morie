"""Tests for rgmatefp.rangayyan_maternal_ecg_filter."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_maternal_ecg_filter


def test_rgmatefp_basic():
    """Test basic functionality."""
    abd = np.random.default_rng(42).normal(0.0, 1.0, 40)
    thor = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_maternal_ecg_filter(abd, thor)
    assert isinstance(result, dict)
    assert "fetal" in result


def test_rgmatefp_edge():
    """Test edge cases."""
    abd = np.random.default_rng(42).normal(0.0, 1.0, 40)
    thor = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_maternal_ecg_filter(abd, thor)
    assert isinstance(result, dict)
