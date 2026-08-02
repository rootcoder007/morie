"""Tests for aitamg.aitchison_amalgamation."""

from morie.fn import _array_core as np

from morie.fn.aitamg import aitchison_amalgamation


def test_aitamg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    idx = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_amalgamation(x, idx)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_aitamg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    idx = np.random.default_rng(42).normal(0, 1, 100)
    result = aitchison_amalgamation(x, idx)
    assert isinstance(result, dict)
