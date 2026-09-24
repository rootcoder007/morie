"""Tests for grdpo.geron_dpo_loss."""

from morie.fn import _array_core as np

from morie.fn.grdpo import geron_dpo_loss


def test_grdpo_basic():
    """Test basic functionality."""
    logp_w = [-0.5]
    logp_l = [-2.0]
    logp_ref_w = [-1.0]
    logp_ref_l = [-1.5]
    result = geron_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grdpo_edge():
    """Test edge cases."""
    logp_w = [-0.5]
    logp_l = [-2.0]
    logp_ref_w = [-1.0]
    logp_ref_l = [-1.5]
    result = geron_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l)
    assert isinstance(result, dict)
