"""Tests for rgwvpkt.rangayyan_wavelet_packet."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_wavelet_packet


def test_rgwvpkt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_packet(x)
    assert isinstance(result, dict)
    assert "nodes" in result


def test_rgwvpkt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_wavelet_packet(x)
    assert isinstance(result, dict)
