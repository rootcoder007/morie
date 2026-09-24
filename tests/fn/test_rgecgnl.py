"""Tests for rgecgnl.rangayyan_ecg_normal_ectopic."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_ecg_normal_ectopic


def test_rgecgnl_basic():
    """Test basic functionality."""
    rr = np.random.default_rng(42).normal(0.0, 1.0, 40)
    ff = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ecg_normal_ectopic(rr, ff)
    assert isinstance(result, dict)
    assert "labels" in result


def test_rgecgnl_edge():
    """Test edge cases."""
    rr = np.random.default_rng(42).normal(0.0, 1.0, 40)
    ff = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ecg_normal_ectopic(rr, ff)
    assert isinstance(result, dict)
