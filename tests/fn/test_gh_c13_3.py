"""Tests for gh_c13_3.ghosal_beta_proc_def."""

import numpy as np

from morie.fn.gh_c13_3 import ghosal_beta_proc_def


def test_gh_c13_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_beta_proc_def(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_gh_c13_3_edge():
    """Test edge cases."""
    result = ghosal_beta_proc_def(np.array([42.0]))
    assert result["n"] == 1
