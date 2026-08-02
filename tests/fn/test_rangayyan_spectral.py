"""Rangayyan spectral template-B repairs.

Anchored on exact spectral identities (Parseval, Wiener-Khinchin, the
AR spectrum of a known AR process) rather than on self-consistency."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgacf import rangayyan_acf_estimate
from morie.fn.rgarsp import rangayyan_ar_spectrum
from morie.fn.rgbwbnd import rangayyan_bandwidth
from morie.fn.rgperio import rangayyan_periodogram
from morie.fn.rgpsdacf import rangayyan_psd_to_acf
from morie.fn.rgwelch import rangayyan_welch_psd
from morie.fn.rgyw import rangayyan_yule_walker


def test_acf_lag_zero_is_the_mean_square_and_divisors_differ():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    out = rangayyan_acf_estimate(x, max_lag=20)
    # R(0) with divisor N is exactly the mean square
    assert out["acf_biased"][0] == pytest.approx(np.mean(x**2))
    assert out["acf_unbiased"][0] == pytest.approx(np.mean(x**2))
    # they diverge as the lag grows: N/(N-m) inflation
    m = 20
    assert out["acf_unbiased"][m] == pytest.approx(
        out["acf_biased"][m] * 500 / (500 - m)
    )
    # white noise: near-zero correlation away from lag 0
    assert abs(out["acf_unbiased"][1:]).max() < 0.2 * out["acf_unbiased"][0]
    with pytest.raises(ValueError):
        rangayyan_acf_estimate(x, max_lag=1000)


def test_periodogram_satisfies_parseval():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(512)
    out = rangayyan_periodogram(x, fs=100.0)
    # Parseval: summing the two-sided |X|^2/N^2 recovers the mean square
    X = np.fft.fft(x)
    assert float(np.sum(np.abs(X) ** 2) / x.size**2) == pytest.approx(np.mean(x**2))
    assert out["total_power"] == pytest.approx(np.mean(x**2))
    # a pure tone puts its power at the right bin
    n = np.arange(512)
    tone = np.sin(2 * np.pi * 10.0 * n / 100.0)
    p = rangayyan_periodogram(tone, fs=100.0)
    assert p["freqs"][int(np.argmax(p["psd"]))] == pytest.approx(10.0, abs=0.3)
    with pytest.raises(ValueError):
        rangayyan_periodogram(x, fs=0.0)


def test_welch_reduces_variance_and_normalises_the_window():
    rng = np.random.default_rng(2)
    # white noise of known variance: the PSD should sit near sigma^2/fs
    x = rng.standard_normal(4096) * 2.0
    w = rangayyan_welch_psd(x, fs=1.0, nperseg=256)
    p = rangayyan_periodogram(x, fs=1.0)
    # Welch is far smoother than the periodogram
    assert np.std(w["psd"][1:]) / np.mean(w["psd"][1:]) < \
        np.std(p["psd"][1:]) / np.mean(p["psd"][1:])
    assert w["n_segments"] > 1
    # the window normalisation keeps the level right: without U the
    # Hann window would bias the estimate low by ~2.7x
    assert w["U"] == pytest.approx(np.mean(np.hanning(256) ** 2))
    assert np.mean(w["psd"][1:]) == pytest.approx(4.0, rel=0.3)  # sigma^2 = 4
    box = rangayyan_welch_psd(x, nperseg=256, window="boxcar")
    assert box["U"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rangayyan_welch_psd(x, window="triangular")


def test_yule_walker_recovers_a_known_ar_process_and_stays_stable():
    rng = np.random.default_rng(3)
    # AR(2): x[n] = 0.75 x[n-1] - 0.5 x[n-2] + e, so a = (-0.75, +0.5)
    n = 4000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.75 * x[t - 1] - 0.5 * x[t - 2] + e[t]
    out = rangayyan_yule_walker(x, order=2)
    assert out["a"][0] == pytest.approx(-0.75, abs=0.06)
    assert out["a"][1] == pytest.approx(0.5, abs=0.06)
    assert out["sigma2"] == pytest.approx(1.0, rel=0.15)
    assert out["stable"] is True  # biased ACF guarantees this
    with pytest.raises(ValueError):
        rangayyan_yule_walker(x[:2], order=5)


def test_ar_spectrum_peaks_where_the_process_resonates():
    rng = np.random.default_rng(4)
    n = 4000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.75 * x[t - 1] - 0.5 * x[t - 2] + e[t]
    out = rangayyan_ar_spectrum(x, order=2, fs=100.0)
    assert out["stable"] is True
    # the AR(2) pole angle gives the resonance: theta = arccos(a1/(2 sqrt(a2)))
    roots = np.roots([1.0, -0.75, 0.5])
    f_expected = float(np.abs(np.angle(roots[0])) / (2 * np.pi) * 100.0)
    f_peak = float(out["freqs"][int(np.argmax(out["psd"]))])
    assert f_peak == pytest.approx(f_expected, abs=2.0)
    with pytest.raises(ValueError):
        rangayyan_ar_spectrum(x, order=2, fs=-1.0)


def test_wiener_khinchin_roundtrip_and_bandwidth_criteria():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(256)
    p = rangayyan_periodogram(x)
    back = rangayyan_psd_to_acf(p["psd"], p["freqs"])
    direct = rangayyan_acf_estimate(x, max_lag=10, biased=True)["acf_biased"]
    # the transform pair agrees with the direct (circular) ACF at lag 0
    assert back["r0"] == pytest.approx(direct[0], rel=0.05)
    assert np.all(np.isreal(back["acf"]))
    with pytest.raises(ValueError):
        rangayyan_psd_to_acf([-1.0, 2.0])
    # bandwidth: on a narrow peak the two criteria differ substantially
    f = np.linspace(0, 50, 501)
    psd = np.exp(-((f - 10.0) ** 2) / (2 * 0.5**2)) + 0.001
    b3 = rangayyan_bandwidth(psd, f, "3dB")
    b99 = rangayyan_bandwidth(psd, f, "99")
    assert b3["f_peak"] == pytest.approx(10.0, abs=0.2)
    assert b99["bandwidth"] > b3["bandwidth"]
    with pytest.raises(ValueError):
        rangayyan_bandwidth(psd, f, "half")
