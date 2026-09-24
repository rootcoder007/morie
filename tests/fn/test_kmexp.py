"""Tests for kmexp.kamath_memorization_exposure."""

from morie.fn import _array_core as np

from morie.fn.kmexp import kamath_memorization_exposure


def test_kmexp_basic():
    """Test basic functionality."""
    canary_ll = -0.1
    candidate_lls = [-2.0, -3.0, -0.5]
    result = kamath_memorization_exposure(canary_ll, candidate_lls)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmexp_edge():
    """Test edge cases."""
    canary_ll = -0.1
    candidate_lls = [-2.0, -3.0, -0.5]
    result = kamath_memorization_exposure(canary_ll, candidate_lls)
    assert isinstance(result, dict)
