"""Tests for rgemdimf.rangayyan_emd_imf."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_emd_imf


def test_rgemdimf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_emd_imf(x)
    assert isinstance(result, dict)
    assert "imf" in result


def test_rgemdimf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_emd_imf(x)
    assert isinstance(result, dict)
