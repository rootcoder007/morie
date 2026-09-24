"""Tests for grptq.geron_static_ptq."""

from morie.fn import _array_core as np

from morie.fn.grptq import geron_static_ptq


def test_grptq_basic():
    """Test basic functionality."""
    model = [lambda a: 2 * a, lambda a: a + 1]
    calibration_data = [[1.0], [2.0]]
    result = geron_static_ptq(model, calibration_data)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grptq_edge():
    """Test edge cases."""
    model = [lambda a: 2 * a, lambda a: a + 1]
    calibration_data = [[1.0], [2.0]]
    result = geron_static_ptq(model, calibration_data)
    assert isinstance(result, dict)
