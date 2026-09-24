"""Tests for otws2.ot_wasserstein_p_1d."""

from morie.fn import _array_core as np

from morie.fn.otws2 import ot_wasserstein_p_1d


def test_otws2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ot_wasserstein_p_1d(x, y)
    assert isinstance(result, dict)
    assert "Wp" in result


def test_otws2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ot_wasserstein_p_1d(x, y)
    assert isinstance(result, dict)
