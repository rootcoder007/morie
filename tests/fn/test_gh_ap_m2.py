"""Tests for gh_ap_m2.ghosal_gibbs_sampler."""

import math

from morie.fn import _array_core as np
from morie.fn.gh_ap_m2 import ghosal_gibbs_sampler


def test_gh_ap_m2_basic():
    """Test basic functionality."""
    result = ghosal_gibbs_sampler(rho=0.6, n_draws=2000, seed=42)
    assert "estimate" in result
    assert "target_rho" in result
    assert "gap" in result
    assert math.isfinite(result["estimate"])
    assert result["target_rho"] == 0.6
    assert math.isfinite(result["gap"])


def test_gh_ap_m2_edge():
    """Test edge cases."""
    # rho = 0 is the boundary of the sqrt
    result = ghosal_gibbs_sampler(rho=0.0, n_draws=100, seed=0)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["target_rho"] == 0.0
