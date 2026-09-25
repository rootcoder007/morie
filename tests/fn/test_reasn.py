"""Tests for reasn.py - Reassigned STFT."""

from morie.fn import _array_core as np

from morie.fn.reasn import reasn, reassigned_stft


def test_reasn_returns_descriptive_result():
    x = np.sin(np.linspace(0, 4 * np.pi, 512))
    result = reassigned_stft(x, fs=512.0, nperseg=64)
    assert result.name == "reassigned_stft"
    assert "spectrogram" in result.extra


def test_reasn_spectrogram_shape():
    x = np.random.default_rng(42).standard_normal(256)
    result = reassigned_stft(x, fs=256.0, nperseg=64)
    spec = result.extra["spectrogram"]
    assert spec.ndim == 2


def test_reasn_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = reasn(x, nperseg=32)
    assert result.name == "reassigned_stft"


def test_an_impulse_is_reassigned_to_its_instant():
    """Auger & Flandrin (1995): t_hat = t + Re(X_th / X_h). For an
    impulse at t0 every bin with energy moves exactly to t0."""
    import pytest

    x = np.zeros(256)
    x[100] = 1.0
    r = reassigned_stft(x, fs=256.0, nperseg=64)
    P = np.asarray(r.extra["spectrogram"]).tolist()
    RT = np.asarray(r.extra["reassigned_times"]).tolist()
    for j in range(len(P[0])):
        if max(P[i][j] for i in range(len(P))) > 1e-9:
            for i in range(len(P)):
                assert RT[i][j] == pytest.approx(100 / 256.0, abs=1e-12)


def test_a_tone_on_the_grid_is_reassigned_to_its_frequency():
    """f_hat = f - Im(X_dh / X_h) / (2 pi). For a complex tone on a DFT
    bin the window-derivative identity X_dh = i (w - w0) X_h is exact
    over the frame, so neighbouring bins move onto f0."""
    import cmath
    import math

    import pytest

    f0 = 8 * 256.0 / 64
    xs = [cmath.exp(2j * math.pi * f0 * k / 256.0) for k in range(256)]
    r = reassigned_stft(xs, fs=256.0, nperseg=64)
    RF = np.asarray(r.extra["reassigned_freqs"]).tolist()
    for i in (7, 8, 9):
        assert RF[i][1] == pytest.approx(f0, abs=1e-9)
