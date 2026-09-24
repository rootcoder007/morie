"""Tests for information_theory_mackay28e22.information_theory_mackay_chapter_28_equation_22."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay28e22 import information_theory_mackay_chapter_28_equation_22


def test_information_theory_mackay28e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_theory_mackay_chapter_28_equation_22(x, t)
    assert isinstance(result, dict)
    assert "logevidence" in result


def test_information_theory_mackay28e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_theory_mackay_chapter_28_equation_22(x, t)
    assert isinstance(result, dict)
