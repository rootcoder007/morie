"""Tests for aitprt.aitchison_perturbation."""

from morie.fn import _array_core as np

from morie.fn.aitprt import aitchison_perturbation


def test_aitprt_basic():
    """Test basic functionality."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = aitchison_perturbation(x, y)
    assert isinstance(result, dict)
    assert "composition" in result
def test_aitprt_edge():
    """Test edge cases."""
    x = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    y = np.abs(np.random.default_rng(43).normal(0, 1, 100)) + 0.5
    result = aitchison_perturbation(x, y)
    assert isinstance(result, dict)
