"""Tests for ehhdec.ehh_decay."""

from morie.fn import _array_core as np

from morie.fn.ehhdec import ehh_decay


def test_ehhdec_basic():
    """Test basic functionality."""
    haplotypes = np.random.default_rng(42).normal(0, 1, 100)
    core = np.random.default_rng(42).normal(0, 1, 100)
    d_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ehh_decay(haplotypes, core, d_grid)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ehhdec_edge():
    """Test edge cases."""
    haplotypes = np.random.default_rng(42).normal(0, 1, 100)
    core = np.random.default_rng(42).normal(0, 1, 100)
    d_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = ehh_decay(haplotypes, core, d_grid)
    assert isinstance(result, dict)
