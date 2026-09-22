"""Tests for hmfmn.geron_fashion_mnist."""

from morie.fn import _array_core as np

from morie.fn.hmfmn import geron_fashion_mnist


def test_hmfmn_basic():
    """Test basic functionality."""
    result = geron_fashion_mnist()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmfmn_edge():
    """Test edge cases."""
    result = geron_fashion_mnist()
    assert isinstance(result, dict)
