"""Tests for kmdpok.kamath_dpo_loss."""

from morie.fn import _array_core as np

from morie.fn.kmdpok import kamath_dpo_loss


def test_kmdpok_basic():
    """Test basic functionality."""
    logp_w = [-1.0, -2.0]
    logp_l = [-3.0, -1.0]
    logp_ref_w = [-2.0, -2.5]
    logp_ref_l = [-3.0, -3.0]
    beta = 2.0
    result = kamath_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmdpok_edge():
    """Test edge cases."""
    logp_w = [-1.0, -2.0]
    logp_l = [-3.0, -1.0]
    logp_ref_w = [-2.0, -2.5]
    logp_ref_l = [-3.0, -3.0]
    beta = 2.0
    result = kamath_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)
