"""Tests for hetero.htmt_ratio."""

import math

from morie.fn import _array_core as np

from morie.fn.hetero import htmt_ratio


def test_hetero_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 6
    X = rng.normal(0, 1, (n, p))
    construct_assignment = [0, 0, 1, 1, 2, 2]
    result = htmt_ratio(X, construct_assignment)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == n
    assert len(result["htmt"]) == 3
    assert result["discriminant_validity"] in (0, 1)
    assert result["threshold"] == 0.85


def test_hetero_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (3, 4))
    construct_assignment = [0, 0, 1, 1]
    result = htmt_ratio(X, construct_assignment)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == 3
    assert len(result["htmt"]) == 1
    assert result["discriminant_validity"] in (0, 1)
