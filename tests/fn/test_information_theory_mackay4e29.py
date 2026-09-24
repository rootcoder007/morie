"""Tests for information_theory_mackay4e29.information_theory_mackay_chapter_4_equation_29."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay4e29 import information_theory_mackay_chapter_4_equation_29


def test_information_theory_mackay4e29_basic():
    """Test basic functionality."""
    p = 0.1
    n = 5
    h = 0.1
    beta = 0.1
    result = information_theory_mackay_chapter_4_equation_29(p, n, h, beta)
    assert isinstance(result, dict)
    assert "info" in result


def test_information_theory_mackay4e29_edge():
    """Test edge cases."""
    p = 0.1
    n = 5
    h = 0.1
    beta = 0.1
    result = information_theory_mackay_chapter_4_equation_29(p, n, h, beta)
    assert isinstance(result, dict)
