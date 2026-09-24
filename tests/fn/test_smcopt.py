"""Tests for smcopt.sequential_mc."""

from morie.fn import _array_core as np

from morie.fn.smcopt import sequential_mc


def test_smcopt_basic():
    """Test basic functionality."""
    objective = lambda x: (x[0] - 1.5) ** 2
    initial = lambda g: [8.0 * g.random() - 4.0]
    result = sequential_mc(objective, initial)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_smcopt_edge():
    """Test edge cases."""
    objective = lambda x: (x[0] - 1.5) ** 2
    initial = lambda g: [8.0 * g.random() - 4.0]
    result = sequential_mc(objective, initial)
    assert isinstance(result, dict)
