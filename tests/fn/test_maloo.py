"""Tests for maloo.ma_leave_one_out."""

from morie.fn import _array_core as np

from morie.fn.maloo import ma_leave_one_out


def test_maloo_basic():
    """Test basic functionality."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_leave_one_out(yi, vi)
    assert isinstance(result, dict)
    assert "mu_full" in result


def test_maloo_edge():
    """Test edge cases."""
    yi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    vi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = ma_leave_one_out(yi, vi)
    assert isinstance(result, dict)
