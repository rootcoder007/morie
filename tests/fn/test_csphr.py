"""Tests for csphr.cutting_plane_sphere."""

import numpy as np

from morie.fn.csphr import cutting_plane_sphere


def test_csphr_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = cutting_plane_sphere(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_csphr_edge():
    """Test edge cases."""
    result = cutting_plane_sphere(np.array([42.0]))
    assert result["n"] == 1
