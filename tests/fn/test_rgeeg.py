"""Tests for rgeeg.rangayyan_eeg_bands.

Spec: Rangayyan & Krishnan (2024) Sec 4.4.1 "Detection of EEG rhythms",
p.228. Band limits follow the canonical delta/theta/alpha/beta/gamma
definitions the section describes.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgeeg import rangayyan_eeg_bands

FS = 256.0


def _sine(freq, n=4096, fs=FS):
    return np.sin(2 * np.pi * freq * np.arange(n) / fs)


def test_rgeeg_alpha_rhythm_lands_in_the_alpha_band():
    # 10 Hz is the classic posterior alpha rhythm (Sec 4.4.1).
    r = rangayyan_eeg_bands(_sine(10.0), fs=FS, nperseg=1024)
    assert max(r["relative"], key=r["relative"].get) == "alpha"
    assert r["relative"]["alpha"] > 0.8


def test_rgeeg_delta_rhythm_lands_in_the_delta_band():
    r = rangayyan_eeg_bands(_sine(2.0), fs=FS, nperseg=1024)
    assert max(r["relative"], key=r["relative"].get) == "delta"


def test_rgeeg_beta_rhythm_lands_in_the_beta_band():
    r = rangayyan_eeg_bands(_sine(20.0), fs=FS, nperseg=1024)
    assert max(r["relative"], key=r["relative"].get) == "beta"


def test_rgeeg_relative_powers_are_fractions_of_the_total():
    r = rangayyan_eeg_bands(_sine(10.0), fs=FS, nperseg=1024)
    for name, val in r["relative"].items():
        assert 0.0 <= val <= 1.0 + 1e-12
        assert r["absolute"][name] == pytest.approx(val * r["total_power"], rel=1e-6)
    # bands are disjoint and lie inside the analysed range, so they cannot
    # sum to more than the whole
    assert sum(r["relative"].values()) <= 1.0 + 1e-9


def test_rgeeg_requires_a_sampling_rate():
    # Band edges are in Hz, so fs is not optional -- there is no meaningful
    # default that would put 8-13 Hz anywhere sensible.
    with pytest.raises(TypeError):
        rangayyan_eeg_bands(_sine(10.0))
