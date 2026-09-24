"""Tests for information_theory_mackay28e13.information_theory_mackay_chapter_28_equation_13."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay28e13 import information_theory_mackay_chapter_28_equation_13


def test_information_theory_mackay28e13_basic():
    """Test basic functionality."""
    factors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = information_theory_mackay_chapter_28_equation_13(factors)
    assert isinstance(result, dict)
    assert "ratio" in result


def test_information_theory_mackay28e13_edge():
    """Test edge cases."""
    factors = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = information_theory_mackay_chapter_28_equation_13(factors)
    assert isinstance(result, dict)
