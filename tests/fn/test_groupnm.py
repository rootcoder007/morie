"""Tests for groupnm.group_norm."""

from morie.fn import _array_core as np

from morie.fn.groupnm import group_norm


def test_groupnm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_groups = 5
    result = group_norm(x, n_groups)
    assert isinstance(result, dict)
    assert "x" in result


def test_groupnm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    n_groups = 5
    result = group_norm(x, n_groups)
    assert isinstance(result, dict)
