"""Tests for information_theory_mackay29e19.information_theory_mackay_chapter_29_equation_19."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay29e19 import information_theory_mackay_chapter_29_equation_19


def test_information_theory_mackay29e19_basic():
    """Test basic functionality."""
    n = 5
    h = 0.1
    result = information_theory_mackay_chapter_29_equation_19(n, h)
    assert isinstance(result, dict)
    assert "log2rmin" in result


def test_information_theory_mackay29e19_edge():
    """Test edge cases."""
    n = 5
    h = 0.1
    result = information_theory_mackay_chapter_29_equation_19(n, h)
    assert isinstance(result, dict)
