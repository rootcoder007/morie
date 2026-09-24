"""Tests for rgenvgm.rangayyan_envelogram."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsatf import rangayyan_envelogram


def test_rgenvgm_basic():
    """Test basic functionality with explicit R peaks."""
    rng = np.random.default_rng(42)
    # Generate a PCG signal long enough to accommodate several beats
    pcg = rng.normal(0, 1, 1500)
    # Define R-peak locations (sample indices)
    r_peaks = [200, 400, 600, 800, 1000, 1200]
    fs = 1000.0
    result = rangayyan_envelogram(pcg, fs=fs, r_peaks=r_peaks)
    # The function returns a RichResult (dict-like)
    assert isinstance(result, dict)
    # Verify expected keys
    for key in ("envelope", "beats", "M", "beat_length", "fs", "method"):
        assert key in result
    # Envelope length should equal beat_length
    L = result["beat_length"]
    assert len(result["envelope"]) == L
    # Beats matrix should have M rows each of length L
    M = result["M"]
    assert M > 0
    assert len(result["beats"]) == M
    assert all(len(b) == L for b in result["beats"])
    # Sampling frequency should be preserved
    assert result["fs"] == fs


def test_rgenvgm_edge():
    """Test that an insufficient number of R peaks raises an error."""
    rng = np.random.default_rng(42)
    pcg = rng.normal(0, 1, 500)
    # Only one R peak -> should raise ValueError
    with pytest.raises(ValueError):
        rangayyan_envelogram(pcg, fs=1000.0, r_peaks=[100])
