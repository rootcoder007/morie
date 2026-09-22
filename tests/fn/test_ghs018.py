"""Tests for ghs018.ghosal_ch3_tree_splitting_variables."""

from morie.fn import _array_core as np

from morie.fn.ghs018 import ghosal_ch3_tree_splitting_variables


def test_ghs018_basic():
    """Test basic functionality."""
    # Three masses: parent, child0, child1
    parent, c0, c1 = 10.0, 3.0, 7.0
    A_epsilon = (parent, c0, c1)
    result = ghosal_ch3_tree_splitting_variables(A_epsilon)
    assert isinstance(result, dict)
    assert "value" in result
    # Independent computation via the documented formula
    V0_expected = c0 / parent
    V1_expected = c1 / parent
    assert result["value"] == [V0_expected, V1_expected]
    assert result["estimate"] == V0_expected


def test_ghos018_basic_with_epsilon():
    """Test basic functionality with epsilon argument ignored (unused)."""
    parent, c0, c1 = 8.0, 2.5, 5.5
    A_epsilon = (parent, c0, c1)
    epsilon = 1e-6
    result = ghosal_ch3_tree_splitting_variables(A_epsilon, epsilon)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == [c0 / parent, c1 / parent]


def test_ghs018_edge():
    """Test edge case: split fractions summing to 1 (complement_gap = 0)."""
    parent, c0, c1 = 6.0, 2.0, 4.0
    A_epsilon = (parent, c0, c1)
    result = ghosal_ch3_tree_splitting_variables(A_epsilon)
    assert isinstance(result, dict)
    assert "value" in result
    assert "complement_gap" in result
    # children sum equals parent, so gap should be 0
    assert abs(result["complement_gap"]) < 1e-12
    assert result["value"] == [c0 / parent, c1 / parent]


def test_ghs018_edge_mismatch():
    """Test edge case: children sum != parent (complement_gap is None)."""
    parent, c0, c1 = 6.0, 2.0, 3.0  # 2+3 != 6
    A_epsilon = (parent, c0, c1)
    result = ghosal_ch3_tree_splitting_variables(A_epsilon)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["complement_gap"] is None
    assert result["value"] == [c0 / parent, c1 / parent]
