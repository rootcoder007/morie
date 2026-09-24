"""Tests for information_theory_mackay3e31.information_theory_mackay_chapter_3_equation_31."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay3e31 import information_theory_mackay_chapter_3_equation_31


def test_information_theory_mackay3e31_basic():
    """Test basic functionality."""
    num = np.random.default_rng(42).normal(0.0, 1.0, 40)
    den = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_theory_mackay_chapter_3_equation_31(num, den)
    assert isinstance(result, dict)
    assert "ratio" in result


def test_information_theory_mackay3e31_edge():
    """Test edge cases."""
    num = np.random.default_rng(42).normal(0.0, 1.0, 40)
    den = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_theory_mackay_chapter_3_equation_31(num, den)
    assert isinstance(result, dict)
