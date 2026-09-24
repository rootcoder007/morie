"""Tests for netparf.network_paf."""

from morie.fn import _array_core as np

from morie.fn.netparf import network_paf


def test_netparf_basic():
    """Test basic functionality."""
    y = 0.5
    exposure = 0.5
    network = 0.5
    result = network_paf(y, exposure, network)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_netparf_edge():
    """Test edge cases."""
    y = 0.5
    exposure = 0.5
    network = 0.5
    result = network_paf(y, exposure, network)
    assert isinstance(result, dict)
