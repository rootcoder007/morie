"""Tests for pmf_sd.pmf_sd."""

from morie.fn import _array_core as np

from morie.fn.pmf_sd import (
    pmf_sd,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e31_basic():
    """Test basic functionality."""
    values = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = pmf_sd(values, probs)
    assert isinstance(result, dict)
    assert "sd" in result


def test_david_j_morin_probability_for_the_enthusiastic_beginner5e31_edge():
    """Test edge cases."""
    values = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = pmf_sd(values, probs)
    assert isinstance(result, dict)
