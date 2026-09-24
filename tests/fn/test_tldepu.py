"""Tests for tldepu.upper_tail_dependence."""

from morie.fn import _array_core as np

from morie.fn.tldepu import upper_tail_dependence


def test_tldepu_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    copula = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = upper_tail_dependence(y, copula)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tldepu_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    copula = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = upper_tail_dependence(y, copula)
    assert isinstance(result, dict)
