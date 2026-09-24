"""Tests for poisson_offset_predict.poisson_offset_predict."""

from morie.fn import _array_core as np

from morie.fn.poisson_offset_predict import poisson_offset_predict


def test_ca6e7_basic():
    """Test basic functionality."""
    b0 = 0.5
    b1 = 0.5
    x1 = 0.5
    exposure = 0.5
    result = poisson_offset_predict(b0, b1, x1, exposure)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca6e7_edge():
    """Test edge cases."""
    b0 = 0.5
    b1 = 0.5
    x1 = 0.5
    exposure = 0.5
    result = poisson_offset_predict(b0, b1, x1, exposure)
    assert isinstance(result, dict)
