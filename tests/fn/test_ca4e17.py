"""Tests for ca4e17.ca_chapter_4_equation_17."""

import math

from morie.fn import _array_core as np

from morie.fn.ca4e17 import ca_chapter_4_equation_17


def test_ca4e17_basic():
    """Test basic functionality."""
    b = 0.5
    se = 0.1
    result = ca_chapter_4_equation_17(b, se)
    payload = result.payload
    assert isinstance(payload, dict)
    assert "lower" in payload
    assert "upper" in payload
    assert math.isfinite(payload["lower"])
    assert math.isfinite(payload["upper"])
    assert payload["lower"] < b < payload["upper"]


def test_ca4e17_edge():
    """Test edge cases."""
    b = 1.0
    se = 0.01
    result = ca_chapter_4_equation_17(b, se)
    payload = result.payload
    assert isinstance(payload, dict)
    assert "lower" in payload
    assert "upper" in payload
    assert math.isfinite(payload["lower"])
    assert math.isfinite(payload["upper"])
