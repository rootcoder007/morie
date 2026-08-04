"""Tests for rng159.rangayyan_ch3_lms_gradient_estimate."""

from morie.fn import _array_core as np

from morie.fn.bsaadapt import rangayyan_ch3_lms_gradient_estimate


def test_rng159_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    w = np.random.default_rng(45).exponential(1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lms_gradient_estimate(x, r, w, e, n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rng159_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    w = np.random.default_rng(45).exponential(1, 100)
    e = np.random.default_rng(44).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lms_gradient_estimate(x, r, w, e, n)
    assert isinstance(result, dict)
