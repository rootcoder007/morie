"""Tests for hetmix.heterogeneous_mixing."""

import math

from morie.fn import _array_core as np

from morie.fn.hetmix import heterogeneous_mixing


def test_hetmix_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    contact_matrix = rng.uniform(0, 1, (10, 10))
    gamma = 1.0
    result = heterogeneous_mixing(contact_matrix, gamma)
    assert isinstance(result, dict)
    assert "R0" in result
    assert "stable_distribution" in result
    assert len(result["stable_distribution"]) == 10
    assert math.isfinite(result["R0"])
    assert result["R0"] >= 0


def test_hetmix_edge():
    """Test edge cases."""
    rng = np.random.default_rng(123)
    contact_matrix = rng.uniform(0, 1, (2, 2))
    gamma = [0.5, 2.0]
    result = heterogeneous_mixing(contact_matrix, gamma)
    assert isinstance(result, dict)
    assert "R0" in result
    assert "stable_distribution" in result
    assert len(result["stable_distribution"]) == 2
    assert math.isfinite(result["R0"])
    assert result["R0"] >= 0
