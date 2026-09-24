"""Tests for gh_ap_m1.ghosal_mh_sampler."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_m1 import ghosal_mh_sampler


def test_gh_ap_m1_basic():
    """Test basic functionality."""
    result = ghosal_mh_sampler(n_draws=2000, seed=42)
    assert "estimate" in result
    assert "mean" in result
    assert "accept_rate" in result
    assert isinstance(result["estimate"], float)
    assert isinstance(result["mean"], float)
    assert 0.0 <= result["accept_rate"] <= 1.0


def test_gh_ap_m1_edge():
    """Test edge cases."""
    result = ghosal_mh_sampler(n_draws=10, seed=0)
    assert "estimate" in result
    assert "method" in result
