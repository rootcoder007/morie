"""Tests for bndbye.bound_bayes_credible."""

from morie.fn import _array_core as np

from morie.fn.bndbye import bound_bayes_credible


def test_bndbye_basic():
    """Test basic functionality with scalar inputs."""
    phi_hat = 1.0
    half_width = 0.5
    se_phi = 0.2
    result = bound_bayes_credible(phi_hat, half_width, se_phi)
    assert isinstance(result, dict)
    # Check that result contains a recognized key
    assert "estimate" in result or "width_ratio_hpd_over_cs" in result


def test_bndbye_edge():
    """Test edge cases with scalar inputs."""
    phi_hat = 2.0
    half_width = 0.1
    se_phi = 0.05
    result = bound_bayes_credible(phi_hat, half_width, se_phi, level=0.9)
    assert isinstance(result, dict)
    # Verify some structural keys exist
    assert "credible_hpd" in result
    assert "confidence_set" in result
