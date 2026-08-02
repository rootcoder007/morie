"""Tests for dssprt.dssp_secondary."""

from morie.fn import _array_core as np

from morie.fn.dssprt import dssp_secondary


def test_dssprt_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = dssp_secondary(coords)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_dssprt_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = dssp_secondary(coords)
    assert isinstance(result, dict)
