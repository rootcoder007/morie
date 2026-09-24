"""Tests for karpV.genetic_programming."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.karpV import genetic_programming


def test_karpV_basic():
    """Test basic functionality."""
    def fitness(tree):
        return 0.0
    result = genetic_programming(fitness=fitness, gens=2, pop_size=5, seed=1)
    assert isinstance(result, dict)
    assert "best_raw" in result
    assert "best_string" in result
    assert "history" in result
    assert math.isfinite(result["best_raw"])


def test_karpV_edge():
    """Test edge cases."""
    def fitness(tree):
        return 0.0
    with pytest.raises(ValueError):
        genetic_programming(fitness=fitness, max_depth_init=1)
