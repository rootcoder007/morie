"""Tests for aitip.aitchison_inner_product."""

from morie.fn import _array_core as np

from morie.fn.aitip import aitchison_inner_product


def test_aitip_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = aitchison_inner_product(x, y)
    assert isinstance(result, dict)
    assert "inner" in result
def test_aitip_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = aitchison_inner_product(x, y)
    assert isinstance(result, dict)
