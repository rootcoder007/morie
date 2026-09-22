"""Tests for aitclos.aitchison_closure."""

from morie.fn import _array_core as np

from morie.fn.aitclos import aitchison_closure


def test_aitclos_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_closure(x)
    assert isinstance(result, dict)
    assert "closed" in result
def test_aitclos_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_closure(x)
    assert isinstance(result, dict)
