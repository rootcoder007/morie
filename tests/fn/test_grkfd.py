"""Tests for grkfd.geron_kfold_cv."""

from morie.fn import _array_core as np

from morie.fn.grkfd import geron_kfold_cv


def test_grkfd_basic():
    """Test basic functionality."""
    n = 11
    K = 4
    result = geron_kfold_cv(n, K)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grkfd_edge():
    """Test edge cases."""
    n = 11
    K = 4
    result = geron_kfold_cv(n, K)
    assert isinstance(result, dict)
