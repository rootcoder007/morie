"""Tests for rng202.rangayyan_ch4_ccf_discrete_with_delay."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_ch4_ccf_discrete_with_delay


def test_rng202_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_ccf_discrete_with_delay(x, y)
    assert isinstance(result, dict)
    assert "ccf" in result


def test_rng202_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_ccf_discrete_with_delay(x, y)
    assert isinstance(result, dict)
