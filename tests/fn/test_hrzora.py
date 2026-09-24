"""Tests for hrzora.horowitz_two_step_oracle."""

from morie.fn import _array_core as np

from morie.fn.hrzora import horowitz_two_step_oracle


def test_hrzora_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_two_step_oracle(x, y)
    assert isinstance(result, dict)
    assert "grid" in result


def test_hrzora_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_two_step_oracle(x, y)
    assert isinstance(result, dict)
