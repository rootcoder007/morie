"""Tests for kmchin.kamath_chinchilla_compute_optimal."""

from morie.fn import _array_core as np

from morie.fn.kmchin import kamath_chinchilla_compute_optimal


def test_kmchin_basic():
    """Test basic functionality."""
    compute_budget = 12000000000.0
    result = kamath_chinchilla_compute_optimal(compute_budget)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmchin_edge():
    """Test edge cases."""
    compute_budget = 12000000000.0
    result = kamath_chinchilla_compute_optimal(compute_budget)
    assert isinstance(result, dict)
