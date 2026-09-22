"""Tests for epimet.epinow2."""

from morie.fn import _array_core as np

from morie.fn.epimet import epinow2


def test_epimet_basic():
    """Test basic functionality."""
    incidence = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    gen_int = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = epinow2(incidence, gen_int)
    assert isinstance(result, dict)
    assert "rt" in result
def test_epimet_edge():
    """Test edge cases."""
    incidence = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    gen_int = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = epinow2(incidence, gen_int)
    assert isinstance(result, dict)
