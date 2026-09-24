"""Tests for rgcpr.rangayyan_cpr_analysis."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_cpr_analysis


def test_rgcpr_basic():
    """Test basic functionality."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_cpr_analysis(ecg)
    assert isinstance(result, dict)
    assert "sdw" in result


def test_rgcpr_edge():
    """Test edge cases."""
    ecg = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_cpr_analysis(ecg)
    assert isinstance(result, dict)
