"""Tests for rgstf.rangayyan_stft.

Spec: Rangayyan & Krishnan (2024) Sec 8.4.1 "The short-time Fourier
transform", p.438, in Sec 8.4 "Fixed Segmentation", p.438. The STFT trades
time resolution against frequency resolution; a chirp is the standard
demonstration that it tracks a frequency that changes with time.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsaxfrm import rangayyan_stft

FS = 512.0


def test_rgstf_locates_a_stationary_tone():
    t = np.arange(4096) / FS
    r = rangayyan_stft(np.sin(2 * np.pi * 60.0 * t), fs=FS, nperseg=256)
    freqs = np.asarray(r["freqs"], dtype=float)
    Sxx = np.abs(np.asarray(r["Sxx"]))
    # average over time, then find the dominant bin
    assert freqs[int(np.argmax(Sxx.mean(axis=1)))] == pytest.approx(60.0, abs=FS / 256)


def test_rgstf_tracks_a_linear_chirp_upward():
    # frequency sweeps 20 -> 200 Hz; the dominant bin must increase in time
    n = 8192
    t = np.arange(n) / FS
    f0, f1 = 20.0, 200.0
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t**2)
    r = rangayyan_stft(np.sin(phase), fs=FS, nperseg=256)
    freqs = np.asarray(r["freqs"], dtype=float)
    Sxx = np.abs(np.asarray(r["Sxx"]))
    peak = freqs[np.argmax(Sxx, axis=0)]
    early, late = peak[: len(peak) // 4].mean(), peak[-len(peak) // 4 :].mean()
    assert late > early + 50.0


def test_rgstf_axes_match_the_spectrogram_shape():
    r = rangayyan_stft(np.random.default_rng(51).standard_normal(4096), fs=FS, nperseg=256)
    Sxx = np.asarray(r["Sxx"])
    assert Sxx.shape[0] == len(np.asarray(r["freqs"]))
    assert Sxx.shape[1] == len(np.asarray(r["times"]))


def test_rgstf_frequency_axis_stops_at_nyquist():
    r = rangayyan_stft(np.random.default_rng(52).standard_normal(2048), fs=FS, nperseg=256)
    freqs = np.asarray(r["freqs"], dtype=float)
    assert freqs[0] == pytest.approx(0.0)
    assert freqs[-1] == pytest.approx(FS / 2, rel=1e-9)


def test_rgstf_longer_window_gives_finer_frequency_spacing():
    # the time-frequency resolution trade-off of Sec 8.4.1
    x = np.random.default_rng(53).standard_normal(8192)
    short = np.asarray(rangayyan_stft(x, fs=FS, nperseg=128)["freqs"], dtype=float)
    long_ = np.asarray(rangayyan_stft(x, fs=FS, nperseg=512)["freqs"], dtype=float)
    assert (long_[1] - long_[0]) < (short[1] - short[0])
