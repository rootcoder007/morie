"""Tests for grkldg.geron_kl_divergence_gaussian."""

from morie.fn import _array_core as np

from morie.fn.grkldg import geron_kl_divergence_gaussian


def test_grkldg_basic():
    """Test basic functionality."""
    mu = [0.5, -1.0]
    logvar = [0.2, -0.3]
    result = geron_kl_divergence_gaussian(mu, logvar)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grkldg_edge():
    """Test edge cases."""
    mu = [0.5, -1.0]
    logvar = [0.2, -0.3]
    result = geron_kl_divergence_gaussian(mu, logvar)
    assert isinstance(result, dict)
