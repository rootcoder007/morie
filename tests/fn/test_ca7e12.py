"""Tests for ca7e12.ca_chapter_7_equation_12."""

import math

from morie.fn import _array_core as np
from morie.fn.ca7e12 import ca_chapter_7_equation_12


def test_ca7e12_basic():
    """Test basic functionality."""
    result = ca_chapter_7_equation_12(b0=1.0, b1=0.5, x1=2.0, u_0j=0.3)
    assert isinstance(result, dict)
    assert "value" in result.payload
    assert math.isfinite(result.payload["value"])


def test_ca7e12_edge():
    """Test edge cases."""
    result = ca_chapter_7_equation_12(b0=0.0, b1=0.0, x1=0.0, u_0j=0.0)
    assert isinstance(result, dict)
    assert "value" in result.payload
    assert math.isfinite(result.payload["value"])
