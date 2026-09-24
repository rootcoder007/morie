"""Tests for rgvagkn.rangayyan_vag_knee_cartilage."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_vag_knee_cartilage


def test_rgvagkn_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_vag_knee_cartilage(vag, fs)
    assert isinstance(result, dict)
    assert "mean" in result


def test_rgvagkn_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_vag_knee_cartilage(vag, fs)
    assert isinstance(result, dict)
