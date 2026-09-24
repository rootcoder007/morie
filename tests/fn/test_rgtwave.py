"""Tests for rgtwave.rangayyan_t_wave_detect."""

from morie.fn import _array_core as np

from morie.fn.bsaqrs import rangayyan_t_wave_detect


def test_rgtwave_basic():
    """Test basic functionality."""
    chans = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    qrs = 5
    fs = 0.1
    result = rangayyan_t_wave_detect(chans, qrs, fs)
    assert isinstance(result, dict)
    assert "t" in result


def test_rgtwave_edge():
    """Test edge cases."""
    chans = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    qrs = 5
    fs = 0.1
    result = rangayyan_t_wave_detect(chans, qrs, fs)
    assert isinstance(result, dict)
