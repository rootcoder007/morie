"""Tests for joint_density_factorizes.joint_density_factorizes."""

from morie.fn import _array_core as np

from morie.fn.joint_density_factorizes import (
    joint_density_factorizes,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e64_basic():
    """Test basic functionality."""
    grid_x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    density_x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    grid_y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    density_y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = joint_density_factorizes(grid_x, density_x, grid_y, density_y)
    assert isinstance(result, dict)
    assert "total_mass" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner6e64_edge():
    """Test edge cases."""
    grid_x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    density_x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    grid_y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    density_y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = joint_density_factorizes(grid_x, density_x, grid_y, density_y)
    assert isinstance(result, dict)
