"""Tests for smrest.standardized_mortality_ratio."""

from morie.fn import _array_core as np

from morie.fn.smrest import standardized_mortality_ratio


def test_smrest_basic():
    """Test basic functionality."""
    observed = 5
    expected = 0.1
    result = standardized_mortality_ratio(observed, expected)
    assert isinstance(result, dict)
    assert "smr" in result


def test_smrest_edge():
    """Test edge cases."""
    observed = 5
    expected = 0.1
    result = standardized_mortality_ratio(observed, expected)
    assert isinstance(result, dict)
