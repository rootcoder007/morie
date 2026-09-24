"""Tests for grfcn.geron_fcn_upsample."""

from morie.fn import _array_core as np

from morie.fn.grfcn import geron_fcn_upsample


def test_grfcn_basic():
    """Test basic functionality."""
    X = [[1.0, 1.0]]
    W = [[1.0, 1.0, 1.0]]
    result = geron_fcn_upsample(X, W)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grfcn_edge():
    """Test edge cases."""
    X = [[1.0, 1.0]]
    W = [[1.0, 1.0, 1.0]]
    result = geron_fcn_upsample(X, W)
    assert isinstance(result, dict)
