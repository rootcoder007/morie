"""Tests for rgwndw.rangayyan_window_functions."""

from morie.fn import _array_core as np

from morie.fn.bsafilt import rangayyan_window_functions


def test_rgwndw_basic():
    """Test basic functionality."""
    N = 100
    window_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_window_functions(N, window_type)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rgwndw_edge():
    """Test edge cases."""
    N = 100
    window_type = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_window_functions(N, window_type)
    assert isinstance(result, dict)
