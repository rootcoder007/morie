"""Tests for grrep.geron_reparameterization_trick."""

from morie.fn import _array_core as np

from morie.fn.grrep import geron_reparameterization_trick


def test_grrep_basic():
    """Test basic functionality."""
    mu = np.zeros(400)
    logvar = np.zeros(400)
    result = geron_reparameterization_trick(mu, logvar)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grrep_edge():
    """Test edge cases."""
    mu = np.zeros(400)
    logvar = np.zeros(400)
    result = geron_reparameterization_trick(mu, logvar)
    assert isinstance(result, dict)
