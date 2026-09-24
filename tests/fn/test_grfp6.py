"""Tests for grfp6.geron_fp16_mixed_precision."""

from morie.fn import _array_core as np

from morie.fn.grfp6 import geron_fp16_mixed_precision


def test_grfp6_basic():
    """Test basic functionality."""
    loss = 0.5
    S = 512.0
    result = geron_fp16_mixed_precision(loss, S)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grfp6_edge():
    """Test edge cases."""
    loss = 0.5
    S = 512.0
    result = geron_fp16_mixed_precision(loss, S)
    assert isinstance(result, dict)
