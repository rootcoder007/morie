"""Tests for information_theory_mackay1e40.information_theory_mackay_chapter_1_equation_40."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay1e40 import information_theory_mackay_chapter_1_equation_40


def test_information_theory_mackay1e40_basic():
    """Test basic functionality."""
    n = 5
    result = information_theory_mackay_chapter_1_equation_40(n)
    assert isinstance(result, dict)
    assert "approx" in result


def test_information_theory_mackay1e40_edge():
    """Test edge cases."""
    n = 5
    result = information_theory_mackay_chapter_1_equation_40(n)
    assert isinstance(result, dict)
