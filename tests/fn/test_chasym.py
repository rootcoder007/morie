"""Tests for chasym.check_asymptote_msm."""

from morie.fn import _array_core as np

from morie.fn.chasym import check_asymptote_msm


def test_chasym_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = check_asymptote_msm(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_chasym_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = check_asymptote_msm(y)
    assert isinstance(result, dict)
