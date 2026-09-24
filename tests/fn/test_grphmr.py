"""Tests for grphmr.graphormer."""

from morie.fn import _array_core as np

from morie.fn.grphmr import graphormer


def test_grphmr_basic():
    """Test basic functionality."""
    H = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    WQ = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WK = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WV = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    bias = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = graphormer(H, WQ, WK, WV, bias)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_grphmr_edge():
    """Test edge cases."""
    H = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    WQ = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WK = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    WV = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    bias = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = graphormer(H, WQ, WK, WV, bias)
    assert isinstance(result, dict)
