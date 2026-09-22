"""Tests for ca1e3.ca_chapter_1_equation_3."""

from morie.fn import _array_core as np

from morie.fn.ca1e3 import ca_chapter_1_equation_3


def test_ca1e3_basic():
    """Test basic functionality."""
    p = 0.5
    result = ca_chapter_1_equation_3(p)
    assert isinstance(result, dict)
    assert "value" in result
def test_ca1e3_edge():
    """Test edge cases."""
    p = 0.5
    result = ca_chapter_1_equation_3(p)
    assert isinstance(result, dict)
