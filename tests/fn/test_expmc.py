"""Tests for expmc.exponential_mechanism."""

from morie.fn import _array_core as np

from morie.fn.expmc import exponential_mechanism


def test_expmc_basic():
    """Test basic functionality."""
    candidates = np.random.default_rng(42).normal(0, 1, 100)
    utility = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_mechanism(candidates, utility)
    assert isinstance(result, dict)
    assert "selected" in result
def test_expmc_edge():
    """Test edge cases."""
    candidates = np.random.default_rng(42).normal(0, 1, 100)
    utility = np.random.default_rng(42).normal(0, 1, 100)
    result = exponential_mechanism(candidates, utility)
    assert isinstance(result, dict)
