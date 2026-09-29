"""Tests for rssgm.py - Reassigned spectrogram."""

from morie.fn import _array_core as np
from morie.fn.rssgm import reassigned_spectrogram, rssgm


def test_reassigned_returns_result():
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 8 * np.pi, 512)) + rng.standard_normal(512) * 0.1
    result = reassigned_spectrogram(x, fs=256.0, window=64, hop=32)
    assert result.name == "reassigned_spectrogram"
    assert "magnitude" in result.extra
    assert "t_reassigned" in result.extra


def test_reassigned_shapes():
    x = np.random.default_rng(42).standard_normal(256)
    result = reassigned_spectrogram(x, fs=100.0, window=32, hop=16)
    mag = result.extra["magnitude"]
    assert mag.shape[0] > 0
    assert mag.shape[1] > 0


def test_reassigned_alias():
    x = np.random.default_rng(42).standard_normal(256)
    result = rssgm(x, fs=100.0, window=32, hop=16)
    assert result.name == "reassigned_spectrogram"


def test_reassignment_operators_recomputed():
    """t_hat = t + Re(X_th/X_h), f_hat = f - Im(X_dh/X_h)/(2 pi) for one cell,
    with the three windowed DFTs written out by hand."""
    import cmath
    import math

    import pytest

    x = [math.sin(0.7 * k) + 0.3 * math.cos(2.1 * k + 0.4) for k in range(40)]
    fs, L, hop = 8.0, 16, 6
    r = reassigned_spectrogram(x, fs=fs, window=L, hop=hop)
    j, m = 2, 3  # third frame, fourth frequency bin
    seg = x[j * hop : j * hop + L]
    h = [0.5 - 0.5 * math.cos(2 * math.pi * k / L) for k in range(L)]
    E = [cmath.exp(-2j * math.pi * m * k / L) for k in range(L)]
    Xh = sum(seg[k] * h[k] * E[k] for k in range(L))
    Xt = sum(seg[k] * (k - L / 2) / fs * h[k] * E[k] for k in range(L))
    Xd = sum(seg[k] * math.pi * fs / L * math.sin(2 * math.pi * k / L) * E[k] for k in range(L))
    tc = (j * hop + L / 2) / fs
    assert float(r.extra["magnitude"][m][j]) == pytest.approx(abs(Xh), rel=1e-12)
    assert float(r.extra["t_reassigned"][m][j]) == pytest.approx(tc + (Xt / Xh).real, rel=1e-12)
    assert float(r.extra["f_reassigned"][m][j]) == pytest.approx(m * fs / L - (Xd / Xh).imag / (2 * math.pi), rel=1e-12)


def test_complex_tone_is_reassigned_to_its_frequency():
    import cmath
    import math

    import pytest

    f0 = 0.19
    x = [cmath.exp(2j * math.pi * f0 * k) for k in range(96)]
    r = reassigned_spectrogram(x, fs=1.0, window=32, hop=16)
    for m in range(17):
        for j in range(len(r.extra["times"])):
            if float(r.extra["magnitude"][m][j]) > 1.0:
                # exact in continuous time; the sampled derivative window
                # leaves an error of order 1e-5 (1.8e-5 here)
                assert float(r.extra["f_reassigned"][m][j]) == pytest.approx(f0, abs=1e-4)
