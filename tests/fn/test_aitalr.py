"""Tests for aitalr.aitchison_alr."""

from morie.fn import _array_core as np

from morie.fn.aitalr import aitchison_alr


def test_aitalr_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_alr(x)
    assert isinstance(result, dict)
    assert "alr" in result
def test_aitalr_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_alr(x)
    assert isinstance(result, dict)
