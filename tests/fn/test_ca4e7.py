"""Tests for ca4e7.ca_chapter_4_equation_7."""

from morie.fn import _array_core as np

from morie.fn.ca4e7 import ca_chapter_4_equation_7


def test_ca4e7_basic():
    """Test basic functionality."""
    p = 0.5
    result = ca_chapter_4_equation_7(p)
    assert isinstance(result, dict)
    assert "value" in result
def test_ca4e7_edge():
    """Test edge cases."""
    p = 0.5
    result = ca_chapter_4_equation_7(p)
    assert isinstance(result, dict)
