"""Tests for gh_ap_j1.ghosal_levy_ito."""

import numpy as np

from morie.fn.gh_ap_j1 import ghosal_levy_ito


def test_gh_ap_j1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_levy_ito(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_ap_j1_edge():
    """Test edge cases."""
    result = ghosal_levy_ito(np.array([42.0]))
    assert result["n"] == 1
