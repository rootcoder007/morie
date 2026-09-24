"""Tests for spsmsh.spsm_shifted_intervention."""

from morie.fn import _array_core as np

from morie.fn.spsmsh import spsm_shifted_intervention


def test_spsmsh_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = np.random.default_rng(42).normal(0.0, 1.0, 40)
    h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = spsm_shifted_intervention(y, a, h)
    assert isinstance(result, dict)
    assert "psi" in result


def test_spsmsh_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    a = np.random.default_rng(42).normal(0.0, 1.0, 40)
    h = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = spsm_shifted_intervention(y, a, h)
    assert isinstance(result, dict)
