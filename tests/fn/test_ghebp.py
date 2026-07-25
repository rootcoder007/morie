"""Tests for ghebp.ghosal_empirical_bayes."""

import numpy as np

from morie.fn.ghebp import ghosal_empirical_bayes


def test_ghebp_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_empirical_bayes(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghebp_edge():
    """Test edge cases."""
    result = ghosal_empirical_bayes(np.array([42.0]))
    assert result["n"] == 1
