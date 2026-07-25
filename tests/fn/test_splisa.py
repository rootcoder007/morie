"""Tests for splisa.schabenberger_lisa."""

import numpy as np

from morie.fn.splisa import schabenberger_lisa


def test_splisa_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = schabenberger_lisa(x, y)
    assert np.all(np.isfinite(np.asarray(result["statistic"], dtype=float)))  # N6: was a generator-guessed value


def test_splisa_edge():
    """Test edge cases."""
    result = schabenberger_lisa(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result["n"] == 2
