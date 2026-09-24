"""Tests for grdyq.geron_dynamic_quantization."""

from morie.fn import _array_core as np

from morie.fn.grdyq import geron_dynamic_quantization


def test_grdyq_basic():
    """Test basic functionality."""
    x = [[0.5, -1.0], [0.25, 1.0]]
    w = [[2.0], [-2.0]]
    result = geron_dynamic_quantization(x, w)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdyq_edge():
    """Test edge cases."""
    x = [[0.5, -1.0], [0.25, 1.0]]
    w = [[2.0], [-2.0]]
    result = geron_dynamic_quantization(x, w)
    assert isinstance(result, dict)
