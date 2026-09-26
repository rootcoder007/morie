"""Tests for psdmt -- Multitaper PSD."""

from morie.fn import _array_core as np
import pytest

from morie.fn._containers import DescriptiveResult
from morie.fn.psdmt import psdmt


def test_psdmt_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(512)
    result = psdmt(x, fs=1000.0)
    assert isinstance(result, DescriptiveResult)
    assert "psd" in result.extra
    assert "frequencies" in result.extra


def test_psdmt_detects_tone():
    fs = 1000
    t = np.arange(0, 1.0, 1 / fs)
    x = np.sin(2 * np.pi * 100 * t)
    result = psdmt(x, fs=fs)
    freqs = result.extra["frequencies"]
    psd = result.extra["psd"]
    peak = freqs[np.argmax(psd)]
    assert abs(peak - 100) < 20


def test_psdmt_n_tapers():
    x = np.random.default_rng(7).standard_normal(256)
    result = psdmt(x, nw=3.0)
    assert result.extra["n_tapers"] == 5


def test_psdmt_is_a_one_sided_density():
    """Parseval: a one-sided PSD integrates to the signal's variance, so
    every bin except DC (and Nyquist for even nfft) carries the folded
    negative frequency.  The estimate is the eigenvalue-weighted average
    of tapered periodograms, sum lambda_k |X_k|^2 / (fs sum lambda_k),
    recomputed here from the returned tapers' concentrations."""
    from morie.fn._signal_core import dpss
    x = np.random.default_rng(1).standard_normal(257)
    r = psdmt(x, fs=100.0, nw=3.0)
    assert r.value == pytest.approx(float(np.var(x)), rel=0.05)   # 5 tapers: chi2_10 noise
    T, lam = dpss(257, 3.0, Kmax=5, return_ratios=True)
    lam = [float(v) for v in lam.tolist()]
    raw = [0.0] * 129
    for t, w in zip(T.tolist(), lam):
        X = np.fft.rfft(np.asarray(t) * x)
        for k, v in enumerate(np.abs(X).tolist()):
            raw[k] += w * v * v
    want = [v / (100.0 * sum(lam)) * (1.0 if k == 0 else 2.0) for k, v in enumerate(raw)]
    assert r.extra["psd"].tolist() == pytest.approx(want, rel=1e-12)
