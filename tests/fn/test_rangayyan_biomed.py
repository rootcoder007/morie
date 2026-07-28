"""Rangayyan biomedical template-B repairs."""

import numpy as np
import pytest

from morie.fn.rgburg import rangayyan_burg_method
from morie.fn.rgcepsp import rangayyan_cepstrum_pitch
from morie.fn.rgeegsp import rangayyan_eeg_spectral
from morie.fn.rgelast import rangayyan_heart_elasticity
from morie.fn.rgenvgm import rangayyan_envelogram
from morie.fn.rgpdfest import rangayyan_pdf_estimate
from morie.fn.rgrmsnw import rangayyan_rms_noise
from morie.fn.rgtfe import rangayyan_transfer_func_est
from morie.fn.rgtwamx import rangayyan_twa_spectral_mx
from morie.fn.rng017 import rangayyan_ch3_acf_ensemble_estimate
from morie.fn.rng018 import rangayyan_ch3_ensemble_average_function
from morie.fn.rng019 import rangayyan_ch3_time_average_mean
from morie.fn.rng020 import rangayyan_ch3_time_averaged_acf
from morie.fn.rng190 import rangayyan_ch4_pan_tompkins_peak_classification
from morie.fn.rng211 import rangayyan_ch4_average_output_noise_power


def test_ensemble_averaging_improves_snr_by_sqrt_M():
    rng = np.random.default_rng(0)
    T, M = 200, 64
    sig = np.sin(2 * np.pi * np.arange(T) / 50.0)
    reps = sig + rng.standard_normal((M, T)) * 2.0
    out = rangayyan_ch3_ensemble_average_function(reps)
    assert out["snr_gain"] == pytest.approx(8.0)  # sqrt(64)
    # the averaged trace is much closer to the truth than one trial
    assert np.std(out["ensemble_mean"] - sig) < np.std(reps[0] - sig) / 4
    with pytest.raises(ValueError):
        rangayyan_ch3_ensemble_average_function(reps, M=99)


def test_ensemble_and_time_averages_agree_only_under_ergodicity():
    rng = np.random.default_rng(1)
    T, M = 400, 200
    # ergodic: zero-mean white noise, both averages -> 0
    erg = rng.standard_normal((M, T))
    ens = rangayyan_ch3_ensemble_average_function(erg)["ensemble_mean"]
    tim = rangayyan_ch3_time_average_mean(erg)["time_mean"]
    assert abs(ens.mean()) < 0.1
    assert abs(np.mean(tim)) < 0.1
    # non-ergodic: each realisation has its own constant offset, so the
    # time averages scatter while the ensemble mean stays near zero
    offs = rng.standard_normal(M)[:, None] * 5.0
    non = rng.standard_normal((M, T)) + offs
    assert rangayyan_ch3_time_average_mean(non)["spread_across_k"] > 3.0
    assert abs(rangayyan_ch3_ensemble_average_function(non)["ensemble_mean"].mean()) < 1.0
    # ensemble ACF at lag 0 is the mean square at that time
    a = rangayyan_ch3_acf_ensemble_estimate(erg, t1=10, tau=0)
    assert a["acf"] == pytest.approx(np.mean(erg[:, 10] ** 2))
    assert rangayyan_ch3_time_averaged_acf(erg[0], tau=0)["acf"] == pytest.approx(
        np.mean(erg[0] ** 2)
    )
    with pytest.raises(ValueError):
        rangayyan_ch3_acf_ensemble_estimate(erg, t1=10, tau=1000)


def test_burg_recovers_ar2_and_is_always_stable():
    rng = np.random.default_rng(2)
    n = 3000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.75 * x[t - 1] - 0.5 * x[t - 2] + e[t]
    out = rangayyan_burg_method(x, order=2)
    assert out["a"][0] == pytest.approx(-0.75, abs=0.06)
    assert out["a"][1] == pytest.approx(0.5, abs=0.06)
    assert out["stable"] is True
    # every reflection coefficient is bounded by 1, which is what
    # makes stability automatic
    assert np.all(np.abs(out["reflection"]) <= 1.0)
    with pytest.raises(ValueError):
        rangayyan_burg_method(x[:2], order=5)


def test_cepstrum_finds_the_pitch_of_a_synthetic_glottal_train():
    fs, f0 = 8000.0, 120.0
    n = np.arange(4096)
    # impulse train at f0 shaped by a smooth vocal-tract-like envelope
    train = np.zeros(n.size)
    train[:: int(fs / f0)] = 1.0
    env = np.exp(-n / 800.0) * np.sin(2 * np.pi * 900.0 * n / fs)
    x = np.convolve(train, env)[: n.size]
    out = rangayyan_cepstrum_pitch(x, fs=fs, f0_range=(60.0, 400.0))
    assert out["f0"] == pytest.approx(f0, rel=0.05)
    assert out["period_s"] == pytest.approx(1.0 / f0, rel=0.05)
    with pytest.raises(ValueError):
        rangayyan_cepstrum_pitch(x, fs=fs, f0_range=(400.0, 60.0))


def test_eeg_band_powers_land_in_the_right_band():
    fs = 256.0
    t = np.arange(int(8 * fs)) / fs
    # a pure 10 Hz rhythm is alpha (8-13 Hz)
    alpha_sig = np.sin(2 * np.pi * 10.0 * t)
    out = rangayyan_eeg_spectral(alpha_sig, fs=fs)
    assert out["relative"]["alpha"] > 0.8
    assert out["relative"]["delta"] < 0.1
    # a 2 Hz rhythm is delta
    d = rangayyan_eeg_spectral(np.sin(2 * np.pi * 2.0 * t), fs=fs)
    assert d["relative"]["delta"] > 0.8
    # relative powers sum to 1
    assert sum(out["relative"].values()) == pytest.approx(1.0)
    # too low a sampling rate cannot represent beta and is refused
    with pytest.raises(ValueError):
        rangayyan_eeg_spectral(alpha_sig, fs=50.0)


def test_transfer_function_recovers_a_known_filter_with_high_coherence():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(8192)
    # y is x delayed and scaled through a 3-tap FIR
    b = np.array([0.5, 0.3, 0.2])
    y = np.convolve(x, b)[: x.size]
    out = rangayyan_transfer_func_est(x, y, fs=100.0, nperseg=512)
    # noiseless linear system: coherence ~ 1 everywhere
    assert np.median(out["coherence"]) > 0.99
    # and |H| matches the true response
    Htrue = np.abs(np.fft.rfft(b, n=512))
    assert np.median(np.abs(out["magnitude"] - Htrue)) < 0.05
    # adding independent noise to y drops the coherence
    noisy = rangayyan_transfer_func_est(x, y + rng.standard_normal(x.size) * 2.0,
                                        nperseg=512)
    assert np.median(noisy["coherence"]) < np.median(out["coherence"])
    # a single segment would give coherence == 1 vacuously
    with pytest.raises(ValueError):
        rangayyan_transfer_func_est(x[:512], y[:512], nperseg=512)


def test_pan_tompkins_trackers_and_threshold_adapt():
    # a run of large peaks pulls SPKI up; small ones pull NPKI up
    out = rangayyan_ch4_pan_tompkins_peak_classification(
        [1.0] * 5 + [0.05] * 5, SPKI=1.0, NPKI=0.05
    )
    assert out["SPKI"] > out["NPKI"]
    assert out["NPKI"] < out["threshold"] < out["SPKI"]
    # the 1/8 coefficient is exact
    one = rangayyan_ch4_pan_tompkins_peak_classification(
        [2.0], SPKI=1.0, NPKI=0.0, is_signal=[True]
    )
    assert one["SPKI"] == pytest.approx(0.125 * 2.0 + 0.875 * 1.0)
    noise = rangayyan_ch4_pan_tompkins_peak_classification(
        [2.0], SPKI=1.0, NPKI=0.4, is_signal=[False]
    )
    assert noise["NPKI"] == pytest.approx(0.125 * 2.0 + 0.875 * 0.4)
    with pytest.raises(ValueError):
        rangayyan_ch4_pan_tompkins_peak_classification([-1.0])


def test_noise_power_tracks_the_filter_energy():
    f = np.linspace(-50, 50, 2001)
    wide = (np.abs(f) <= 20).astype(float)
    narrow = (np.abs(f) <= 5).astype(float)
    pw = rangayyan_ch4_average_output_noise_power(2.0, wide, f)
    pn = rangayyan_ch4_average_output_noise_power(2.0, narrow, f)
    # four times the bandwidth, four times the output noise
    assert pw["output_power"] == pytest.approx(4 * pn["output_power"], rel=0.02)
    assert pw["noise_equivalent_bw"] == pytest.approx(40.0, rel=0.02)
    with pytest.raises(ValueError):
        rangayyan_ch4_average_output_noise_power(-1.0, wide, f)


def test_rms_noise_and_density_estimates():
    rng = np.random.default_rng(4)
    x = np.r_[rng.standard_normal(200) * 0.1, rng.standard_normal(200) * 3.0]
    out = rangayyan_rms_noise(x, noise_segments=[(0, 200)])
    assert out["rms_noise"] == pytest.approx(0.1, rel=0.25)
    assert out["segments_given"] is True
    assert out["snr_db"] > 10
    # letting signal leak into the "noise" window inflates sigma and
    # collapses the SNR
    leaky = rangayyan_rms_noise(x, noise_segments=[(0, 400)])
    assert leaky["rms_noise"] > out["rms_noise"] * 5
    assert leaky["snr_db"] < out["snr_db"]
    # density estimate integrates to ~1 and reports its bandwidth
    d = rangayyan_pdf_estimate(rng.standard_normal(2000))
    assert d["integrates_to"] == pytest.approx(1.0, abs=0.05)
    assert d["bandwidth"] > 0
    with pytest.raises(ValueError):
        rangayyan_pdf_estimate(x, method="spline")


def test_envelogram_requires_alignment_and_twa_requires_even_beats():
    fs = 1000.0
    rng = np.random.default_rng(5)
    n = 10000
    r = np.arange(500, n - 500, 800)
    pcg = rng.standard_normal(n) * 0.05
    for p in r:  # a burst just after each R peak
        pcg[p + 50 : p + 150] += np.hanning(100) * 2.0
    out = rangayyan_envelogram(pcg, fs=fs, r_peaks=r)
    assert out["M"] >= 5
    # the averaged envelope peaks inside the burst window
    assert 40 <= int(np.argmax(out["envelope"])) <= 200
    with pytest.raises(ValueError):
        rangayyan_envelogram(pcg, fs=fs)  # no alignment supplied
    # TWA: alternating T amplitudes give a 0.5 cyc/beat peak
    ecg = rng.standard_normal(n) * 0.01
    for i, p in enumerate(r):
        ecg[p + 100 : p + 300] += np.hanning(200) * (1.0 + 0.3 * (-1) ** i)
    twa = rangayyan_twa_spectral_mx(ecg, fs=fs, r_peaks=r, n_beats=len(r))
    assert twa["n_beats_used"] % 2 == 0  # even, so 0.5 is an exact bin
    assert twa["alternans_voltage"] > 0
    with pytest.raises(ValueError):
        rangayyan_twa_spectral_mx(ecg, fs=fs, r_peaks=r[:3])


def test_heart_sound_index_refuses_to_fake_calibration():
    fs = 2000.0
    t = np.arange(2000) / fs
    stiff = np.sin(2 * np.pi * 120.0 * t) * np.exp(-t * 20)
    soft = np.sin(2 * np.pi * 40.0 * t) * np.exp(-t * 20)
    hi = rangayyan_heart_elasticity(stiff, fs=fs)
    lo = rangayyan_heart_elasticity(soft, fs=fs)
    # the stiffer sound sits higher in frequency, as the text states
    assert hi["dominant_frequency"] > lo["dominant_frequency"]
    assert hi["spectral_centroid"] > lo["spectral_centroid"]
    # and no fabricated absolute stiffness is reported
    assert hi["calibrated"] is False
    with pytest.raises(ValueError):
        rangayyan_heart_elasticity(stiff, fs=fs, s1_window=(0, 99999))
