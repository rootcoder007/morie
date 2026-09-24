"""Tests for grig.geron_information_gain."""

from morie.fn import _array_core as np

import pytest

from morie.fn.grig import geron_information_gain


def test_grig_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    y = rng_y.integers(0, 2, 100)
    rng_mask = np.random.default_rng(42)
    left_mask = rng_mask.integers(0, 2, 100).astype(bool)
    result = geron_information_gain(y, left_mask)
    assert isinstance(result, dict)
    assert "information_gain" in result
    assert "parent_impurity" in result
    assert "left_impurity" in result
    assert "right_impurity" in result
    assert "m_left" in result
    assert "m_right" in result
    assert "weighted_child_impurity" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result


def test_grig_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        geron_information_gain([0, 1, 0, 1], [True, False, True, False], criterion="bogus")
