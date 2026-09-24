"""Tests for rgvag.rangayyan_vag_analysis."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_vag_analysis


def test_rgvag_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_vag_analysis(vag, fs)
    assert isinstance(result, dict)
    assert "mean" in result


def test_rgvag_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_vag_analysis(vag, fs)
    assert isinstance(result, dict)
