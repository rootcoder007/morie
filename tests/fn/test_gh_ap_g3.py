"""Tests for gh_ap_g3.ghosal_dir_marginal."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_ap_g3 import ghosal_dir_marginal


def test_gh_ap_g3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_dir_marginal(x)
    assert "estimate" in result
    assert "beta_params" in result
    assert "variance" in result
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert result["variance"] >= 0.0


def test_gh_ap_g3_edge():
    """Test edge cases."""
    result = ghosal_dir_marginal(np.array([42.0]), merge_idx=(0,))
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == 1.0
