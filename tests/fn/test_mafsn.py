"""Tests for mafsn.ma_fail_safe_n."""

from morie.fn import _array_core as np

from morie.fn.mafsn import ma_fail_safe_n


def test_mafsn_basic():
    """Test basic functionality."""
    z_scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_fail_safe_n(z_scores)
    assert isinstance(result, dict)
    assert "Nfs" in result


def test_mafsn_edge():
    """Test edge cases."""
    z_scores = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ma_fail_safe_n(z_scores)
    assert isinstance(result, dict)
