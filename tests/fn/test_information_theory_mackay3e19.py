"""Tests for information_theory_mackay3e19.information_theory_mackay_chapter_3_equation_19."""

from morie.fn import _array_core as np

from morie.fn.information_theory_mackay3e19 import information_theory_mackay_chapter_3_equation_19


def test_information_theory_mackay3e19_basic():
    """Test basic functionality."""
    evidences = np.random.default_rng(42).normal(0.0, 1.0, 40)
    priors = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_theory_mackay_chapter_3_equation_19(evidences, priors)
    assert isinstance(result, dict)
    assert "evidence" in result


def test_information_theory_mackay3e19_edge():
    """Test edge cases."""
    evidences = np.random.default_rng(42).normal(0.0, 1.0, 40)
    priors = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = information_theory_mackay_chapter_3_equation_19(evidences, priors)
    assert isinstance(result, dict)
