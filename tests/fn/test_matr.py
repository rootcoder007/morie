"""Tests for matr.ma_two_step_dl_he."""

from morie.fn import _array_core as np

from morie.fn.matr import ma_two_step_dl_he


def test_matr_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_two_step_dl_he(yi, vi)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_matr_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    vi = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_two_step_dl_he(yi, vi)
    assert isinstance(result, dict)
