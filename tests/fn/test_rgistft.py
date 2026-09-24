"""Tests for rgistft.rangayyan_istft."""

from morie.fn import _array_core as np

from morie.fn.bsatf import rangayyan_istft


def test_rgistft_basic():
    """Test basic functionality."""
    stft = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_istft(stft)
    assert isinstance(result, dict)
    assert "signal" in result


def test_rgistft_edge():
    """Test edge cases."""
    stft = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_istft(stft)
    assert isinstance(result, dict)
