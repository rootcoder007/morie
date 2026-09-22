"""Tests for drspl.dr_did_split_sample."""

from morie.fn import _array_core as np

from morie.fn.drspl import dr_did_split_sample


def test_drspl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_split_sample(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_drspl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_split_sample(y, D)
    assert isinstance(result, dict)
