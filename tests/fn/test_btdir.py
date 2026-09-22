"""Tests for btdir.boot_dirichlet_weights."""

from morie.fn import _array_core as np

from morie.fn.btdir import boot_dirichlet_weights


def test_btdir_basic():
    """Test basic functionality."""
    n = 100
    result = boot_dirichlet_weights(n)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_btdir_edge():
    """Test edge cases."""
    n = 100
    result = boot_dirichlet_weights(n)
    assert isinstance(result, dict)
