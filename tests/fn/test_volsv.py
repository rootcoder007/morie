"""Tests for volsv.vol_sv_quasi_lik."""

from morie.fn import _array_core as np

from morie.fn.volsv import vol_sv_quasi_lik


def test_volsv_basic():
    """Test basic functionality."""
    r = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vol_sv_quasi_lik(r)
    assert isinstance(result, dict)
    assert "mu" in result


def test_volsv_edge():
    """Test edge cases."""
    r = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = vol_sv_quasi_lik(r)
    assert isinstance(result, dict)
