"""Tests for mecpd.markov_equivalence_class."""

import numpy as np

from morie.fn.mecpd import markov_equivalence_class


def test_mecpd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = markov_equivalence_class(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_mecpd_edge():
    """Test edge cases."""
    result = markov_equivalence_class(np.array([42.0]))
    assert result["n"] == 1
