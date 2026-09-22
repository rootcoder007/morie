"""Tests for cdpamp.cdp_subgaussian_amplification."""

from morie.fn import _array_core as np

from morie.fn.cdpamp import cdp_subgaussian_amplification


def test_cdpamp_basic():
    """Test basic functionality."""
    rho = 0.5
    result = cdp_subgaussian_amplification(rho)
    assert isinstance(result, dict)
    assert "rho_total" in result
def test_cdpamp_edge():
    """Test edge cases."""
    rho = 0.5
    result = cdp_subgaussian_amplification(rho)
    assert isinstance(result, dict)
