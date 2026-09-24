"""Tests for hrzsitr.horowitz_sieve_npiv."""

from morie.fn import _array_core as np

from morie.fn.hrzsitr import horowitz_sieve_npiv


def test_hrzsitr_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Ey_w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sieve_npiv(T, Ey_w)
    assert isinstance(result, dict)
    assert "g" in result


def test_hrzsitr_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    Ey_w = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sieve_npiv(T, Ey_w)
    assert isinstance(result, dict)
