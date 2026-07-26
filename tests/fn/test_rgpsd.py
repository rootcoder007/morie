"""Tests for rgpsd.rangayyan_psd.

Spec: Rangayyan & Krishnan (2024) Sec 6.3.2 "The periodogram" p.323,
Sec 6.3.3 "The need for averaging PSDs" p.325, Sec 6.3.4 "The use of
windows" p.326; Welch (1967). The identities pinned here are Parseval
(integrated PSD = signal power) and peak location.
"""

import numpy as np
import pytest

from morie.fn.rgpsd import rangayyan_psd


def test_rgpsd_peaks_at_the_sinusoid_frequency():
    fs = 256.0
    t = np.arange(2048) / fs
    r = rangayyan_psd(np.sin(2 * np.pi * 40.0 * t), fs=fs, nperseg=512)
    assert r["peak_freq"] == pytest.approx(40.0, abs=fs / 512)


def test_rgpsd_integrates_to_signal_power():
    # Parseval: the integral of the one-sided PSD is the mean square of the
    # signal. A unit-amplitude sine has mean square 1/2.
    fs = 256.0
    t = np.arange(8192) / fs
    r = rangayyan_psd(np.sin(2 * np.pi * 40.0 * t), fs=fs, nperseg=1024)
    assert r["total_power"] == pytest.approx(0.5, rel=0.05)


def test_rgpsd_white_noise_power_matches_variance():
    fs = 100.0
    x = np.random.default_rng(11).standard_normal(16384)
    r = rangayyan_psd(x, fs=fs, nperseg=1024)
    assert r["total_power"] == pytest.approx(float(np.var(x)), rel=0.05)


def test_rgpsd_is_nonnegative():
    x = np.random.default_rng(12).standard_normal(1024)
    psd = np.asarray(rangayyan_psd(x, fs=50.0)["psd"], dtype=float)
    assert np.all(psd >= 0.0)


def test_rgpsd_survives_numpy2_trapezoid_rename():
    # np.trapz was REMOVED in NumPy 2.0; this returned AttributeError for
    # every caller until the module bound np.trapezoid instead.
    r = rangayyan_psd(np.random.default_rng(13).standard_normal(512), fs=10.0)
    assert np.isfinite(r["total_power"])
