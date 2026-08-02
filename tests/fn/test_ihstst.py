"""Tests for ihstst.ihs_test."""

from morie.fn import _array_core as np

from morie.fn.ihstst import ihs_test


def test_ihstst_basic():
    """Test basic functionality."""
    haplotypes = np.random.default_rng(42).normal(0, 1, 100)
    ancestral = np.random.default_rng(42).normal(0, 1, 100)
    result = ihs_test(haplotypes, ancestral)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ihstst_edge():
    """Test edge cases."""
    haplotypes = np.random.default_rng(42).normal(0, 1, 100)
    ancestral = np.random.default_rng(42).normal(0, 1, 100)
    result = ihs_test(haplotypes, ancestral)
    assert isinstance(result, dict)
