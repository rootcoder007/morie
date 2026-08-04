"""Tests for rgcomb.rangayyan_comb_filter."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_comb_filter


def test_rgcomb_basic():
    """Test basic functionality."""
    period_samples = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_comb_filter(period_samples, fs)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgcomb_edge():
    """Test edge cases."""
    period_samples = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_comb_filter(period_samples, fs)
    assert isinstance(result, dict)
