"""Tests for gh_c14_17.ghosal_disc_rp_rel."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_17 import ghosal_disc_rp_rel


def test_gh_c14_17_basic():
    """Test basic functionality with default (DP reduction) inputs."""
    result = ghosal_disc_rp_rel(d=0.0, theta=1.0)
    assert "estimate" in result
    assert np.isfinite(float(result["estimate"]))
    # At d=0, PY reduces to DP, so estimate should be 1.0.
    assert float(result["estimate"]) == 1.0


def test_gh_c14_17_non_dp():
    """When d != 0, the reduction does not hold and estimate is 0.0."""
    result = ghosal_disc_rp_rel(d=0.5, theta=2.0)
    assert float(result["estimate"]) == 0.0


def test_gh_c14_17_edge():
    """Test edge case: single scalar input still produces an 'estimate' key."""
    result = ghosal_disc_rp_rel(d=0.0, theta=1.0)
    assert "estimate" in result
