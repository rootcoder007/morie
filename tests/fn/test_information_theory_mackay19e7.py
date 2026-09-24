"""Tests for information_theory_mackay19e7.information_theory_mackay_chapter_19_equation_7."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay19e7 import information_theory_mackay_chapter_19_equation_7


def test_information_theory_mackay19e7_basic():
    """Test basic functionality."""
    gamma = 0.1
    result = information_theory_mackay_chapter_19_equation_7(gamma)
    assert isinstance(result, dict)
    assert "onepbeta" in result


def test_information_theory_mackay19e7_edge():
    """Test edge cases."""
    gamma = 0.1
    result = information_theory_mackay_chapter_19_equation_7(gamma)
    assert isinstance(result, dict)
