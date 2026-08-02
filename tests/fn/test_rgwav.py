"""Tests for rgwav.rangayyan_wavelet_denoise.

Spec: Donoho & Johnstone (1994), Biometrika 81(3):425-455 -- the universal
threshold sigma*sqrt(2 log n) and soft thresholding. Application context:
Rangayyan & Krishnan (2024) Sec 8.14, p.493.
"""


from morie.fn import _array_core as np
import pytest

from morie.fn.rgwav import rangayyan_wavelet_denoise

# Without pywt the function documents a moving-average fallback that reports
# no sigma and no threshold, so the wavelet-specific properties below have
# nothing to assert on. Length preservation and error reduction still do.


def test_rgwav_threshold_is_the_universal_threshold():
    # Donoho & Johnstone's universal threshold is sigma * sqrt(2 ln n).
    n = 1024
    x = np.random.default_rng(61).standard_normal(n)
    r = rangayyan_wavelet_denoise(x)
    assert r["threshold"] == pytest.approx(r["sigma"] * np.sqrt(2.0 * np.log(n)), rel=1e-9)


def test_rgwav_reduces_error_on_a_noisy_smooth_signal():
    n = 2048
    t = np.linspace(0.0, 1.0, n)
    clean = np.sin(2 * np.pi * 3.0 * t)
    noisy = clean + 0.4 * np.random.default_rng(62).standard_normal(n)
    out = np.asarray(rangayyan_wavelet_denoise(noisy)["signal"], dtype=float)
    assert np.mean((out - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_rgwav_preserves_length():
    x = np.random.default_rng(63).standard_normal(777)
    out = np.asarray(rangayyan_wavelet_denoise(x)["signal"], dtype=float)
    assert out.size == x.size


def test_rgwav_soft_threshold_shrinks_towards_zero():
    # Soft thresholding never increases magnitude, so a pure-noise record
    # must come back with no more energy than it went in with.
    x = np.random.default_rng(64).standard_normal(1024)
    out = np.asarray(rangayyan_wavelet_denoise(x, mode="soft")["signal"], dtype=float)
    assert np.sum(out**2) <= np.sum(x**2) + 1e-9


def test_rgwav_estimated_sigma_tracks_the_injected_noise():
    # sigma is the MAD-based robust scale estimate of the finest detail level
    n = 4096
    clean = np.sin(2 * np.pi * 2.0 * np.linspace(0, 1, n))
    for s in (0.1, 0.5):
        noisy = clean + s * np.random.default_rng(65).standard_normal(n)
        assert rangayyan_wavelet_denoise(noisy)["sigma"] == pytest.approx(s, rel=0.25)
