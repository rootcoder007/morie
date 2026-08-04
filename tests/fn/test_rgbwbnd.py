"""Tests for rgbwbnd.rangayyan_bandwidth."""

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_bandwidth


def test_rgbwbnd_basic():
    """Test basic functionality."""
    # a PSD is non-negative over an increasing frequency grid and the
    # criterion is one of the two documented strings (Rangayyan Ch. 3);
    # the generator fed noise for all three.
    freqs = [i * 0.5 for i in range(100)]
    psd = [1.0 / (1.0 + ((f - 10.0) / 3.0) ** 2) for f in freqs]
    criterion = "3dB"
    result = rangayyan_bandwidth(psd, freqs, criterion)
    assert isinstance(result, dict)
    # Lorentzian 1/(1+((f-10)/3)^2): half power at f = 10 -+ 3, so the
    # -3 dB band is [7, 13] and the width 6 (to grid resolution 0.5)
    assert abs(result["f_peak"] - 10.0) <= 0.5
    assert abs(result["bandwidth"] - 6.0) <= 1.0


def test_rgbwbnd_edge():
    """Test edge cases."""
    # a PSD is non-negative over an increasing frequency grid and the
    # criterion is one of the two documented strings (Rangayyan Ch. 3);
    # the generator fed noise for all three.
    freqs = [i * 0.5 for i in range(100)]
    psd = [1.0 / (1.0 + ((f - 10.0) / 3.0) ** 2) for f in freqs]
    criterion = "3dB"
    result = rangayyan_bandwidth(psd, freqs, criterion)
    assert isinstance(result, dict)
