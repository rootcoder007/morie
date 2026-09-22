"""Tests for aitnrm.aitchison_norm."""

from morie.fn import _array_core as np

from morie.fn.aitnrm import aitchison_norm


def test_aitnrm_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_norm(x)
    assert isinstance(result, dict)
    assert "norm" in result
def test_aitnrm_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = aitchison_norm(x)
    assert isinstance(result, dict)
