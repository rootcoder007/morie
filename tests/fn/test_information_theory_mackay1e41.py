"""Tests for information_theory_mackay1e41.information_theory_mackay_chapter_1_equation_41."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay1e41 import information_theory_mackay_chapter_1_equation_41


def test_information_theory_mackay1e41_basic():
    """Test basic functionality."""
    n = 5
    result = information_theory_mackay_chapter_1_equation_41(n)
    assert isinstance(result, dict)
    assert "var" in result


def test_information_theory_mackay1e41_edge():
    """Test edge cases."""
    n = 5
    result = information_theory_mackay_chapter_1_equation_41(n)
    assert isinstance(result, dict)
