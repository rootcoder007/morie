"""Tests for rng156.rangayyan_ch3_lms_estimation_error."""

from morie.fn import _array_core as np

from morie.fn.bsaadapt import rangayyan_ch3_lms_estimation_error


def test_rng156_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_lms_estimation_error(x, w, r, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng156_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    r = 10
    n = 100
    result = rangayyan_ch3_lms_estimation_error(x, w, r, n)
    assert isinstance(result, dict)
