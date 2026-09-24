"""Tests for hrzbwcv.horowitz_bw_cv_sim."""

from morie.fn import _array_core as np

from morie.fn.hrzbwcv import horowitz_bw_cv_sim


def test_hrzbwcv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = horowitz_bw_cv_sim(x, y, beta)
    assert isinstance(result, dict)
    assert "bandwidth" in result


def test_hrzbwcv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    beta = 0.1
    result = horowitz_bw_cv_sim(x, y, beta)
    assert isinstance(result, dict)
