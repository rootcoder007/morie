"""Tests for joint_density_factorizes.joint_density_factorizes."""

from morie.fn import _array_core as np

from morie.fn.joint_density_factorizes import (
    joint_density_factorizes,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e64_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_density_factorizes(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e64_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_density_factorizes(x)
    assert isinstance(result, dict)
