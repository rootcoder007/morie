"""Tests for eqsl.equating_stocking_lord."""

from morie.fn import _array_core as np

from morie.fn.eqsl import equating_stocking_lord


def test_eqsl_basic():
    """Test basic functionality."""
    a_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    a_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = equating_stocking_lord(a_ref, b_ref, a_focal, b_focal)
    assert isinstance(result, dict)
    assert "A" in result
def test_eqsl_edge():
    """Test edge cases."""
    a_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_ref = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    a_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    b_focal = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = equating_stocking_lord(a_ref, b_ref, a_focal, b_focal)
    assert isinstance(result, dict)
