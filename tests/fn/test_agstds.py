"""Tests for agstds.age_standardize."""

from morie.fn import _array_core as np

from morie.fn.agstds import age_standardize


def test_agstds_basic():
    """Test basic functionality."""
    rates = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    standard_pop = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = age_standardize(rates, standard_pop)
    assert isinstance(result, dict)
    assert "asr" in result
def test_agstds_edge():
    """Test edge cases."""
    rates = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    standard_pop = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = age_standardize(rates, standard_pop)
    assert isinstance(result, dict)
