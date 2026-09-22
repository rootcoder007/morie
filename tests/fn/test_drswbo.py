"""Tests for drswbo.dr_did_stratified_block."""

from morie.fn import _array_core as np

from morie.fn.drswbo import dr_did_stratified_block


def test_drswbo_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_stratified_block(y, D)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_drswbo_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_did_stratified_block(y, D)
    assert isinstance(result, dict)
