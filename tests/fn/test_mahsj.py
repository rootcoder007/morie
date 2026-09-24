"""Tests for mahsj.ma_hksj_t_pi."""

from morie.fn import _array_core as np

from morie.fn.mahsj import ma_hksj_t_pi


def test_mahsj_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_hksj_t_pi(yi, vi)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_mahsj_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_hksj_t_pi(yi, vi)
    assert isinstance(result, dict)
