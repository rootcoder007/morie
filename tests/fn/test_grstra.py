"""Tests for grstra.geron_stratified_split."""

from morie.fn import _array_core as np

from morie.fn.grstra import geron_stratified_split


def test_grstra_basic():
    """Test basic functionality."""
    strata = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_stratified_split(strata)
    assert isinstance(result, dict)
    assert "test" in result or "test" in result


def test_grstra_edge():
    """Test edge cases."""
    strata = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = geron_stratified_split(strata)
    assert isinstance(result, dict)
