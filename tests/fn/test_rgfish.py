"""Tests for rgfish.rangayyan_fisher_criterion."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_fisher_criterion


def test_rgfish_basic():
    """Test basic functionality."""
    x1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x2 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_fisher_criterion(x1, x2)
    assert isinstance(result, dict)
    assert "j" in result


def test_rgfish_edge():
    """Test edge cases."""
    x1 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x2 = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_fisher_criterion(x1, x2)
    assert isinstance(result, dict)
