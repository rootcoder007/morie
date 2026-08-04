"""Rangayyan correlation, spectral density and the matched filter
(bsacorr).

Expected values are hand-computed from the printed equations, or are
properties the book states: the matched filter's impulse response is the
reversed reference (eq. 4.54), its output is the reference ACF, and its
maximum SNR is 2 E / N0 (eq. 4.46).
"""

import math

import pytest

from morie.fn.bsacorr import (cardioresp, cauchysch, ccfouter, cohere,
                              contproj, csd, dotprod, emgfreq, erpartifact,
                              idft, matchedfilt, mfacf, mfimpeeg, mfimpulse,
                              mfinput, mfmaxsnr, mfnoisein, mfnoiseout,
                              mfoutput, mfpeak, mfpsd, mfratio, mfsnr, mftf,
                              mftfeeg, msc, parseval, pcgsyncavg, psdhz,
                              refpattern, schwarzc, schwarzr, seizcohere,
                              sigenergy, specmoments, specres, syncsum,
                              template, triangle)


def sine(n, cycles, amp=1.0, phase=0.0):
    return [amp * math.sin(2 * math.pi * cycles * i / n + phase)
            for i in range(n)]


# ------------------------------------------------- inner products 4.24-4.29

def test_dotprod_eqs424_425():
    r = dotprod([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert r["dot_product"] == pytest.approx(32.0)
    assert r["gamma"] == pytest.approx(
        32.0 / math.sqrt(14.0 * 77.0))


def test_dotprod_gamma_is_one_for_a_scaled_copy():
    assert dotprod([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])["gamma"] == \
        pytest.approx(1.0)


def test_dotprod_mean_removal_changes_the_answer():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    raw = dotprod(x, y)["gamma"]
    centred = dotprod(x, y, subtract_mean=True)["gamma"]
    assert raw == pytest.approx(10.0 / 14.0)   # both positive, so > 0
    assert centred == pytest.approx(-1.0)  # perfectly anti-correlated


def test_contproj_eq426_carries_the_dt():
    x = [1.0] * 5
    y = [2.0] * 5
    r = contproj(x, y, dt=0.5)
    assert r["discrete_sum"] == pytest.approx(10.0)
    # trapezoid over [0, 2] of a constant 2 is 4
    assert r["theta"] == pytest.approx(4.0)


def test_ccfouter_eq429_is_toeplitz_for_a_stationary_input():
    x = sine(600, 17)
    r = ccfouter(x, x, order=4)
    assert r["toeplitz"] is True
    assert len(r["theta"]) == 4


def test_ccfouter_deviation_grows_with_nonstationarity():
    # the flag rests on a tolerance; the deviation itself is the measure
    flat = ccfouter(sine(600, 17), sine(600, 17), order=4)
    ramp = [v * (1.0 + 4.0 * i / 600) for i, v in enumerate(sine(600, 17))]
    tilted = ccfouter(ramp, ramp, order=4)
    assert tilted["relative_deviation"] > 2 * flat["relative_deviation"]
    assert tilted["toeplitz"] is False


def test_ccfouter_rejects_too_short_a_record():
    with pytest.raises(ValueError):
        ccfouter([1.0, 2.0], [1.0, 2.0], order=5)


# ---------------------------------------------------- PSD, CSD, coherence

def test_csd_eqs430_431_two_routes_agree():
    x = sine(64, 5)
    y = sine(64, 5, phase=0.4)
    r = csd(x, y)
    assert r["agrees"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-6)


def test_csd_of_a_signal_with_itself_is_the_psd():
    x = sine(64, 5)
    r = csd(x, x)
    assert max(abs(v.imag) for v in r["csd"]) < 1e-8


def test_cohere_eq432_refuses_a_single_segment():
    # the book: computed on one observation the coherence is 1 everywhere
    x = sine(64, 5)
    y = sine(64, 5, phase=0.3)
    with pytest.raises(ValueError):
        cohere(x, y, nperseg=64)


def test_cohere_is_high_for_a_linearly_related_pair():
    n = 1024
    x = [sine(n, 13)[i] + 0.2 * sine(n, 97)[i] for i in range(n)]
    y = [2.0 * x[i] for i in range(n)]
    r = cohere(x, y, fs=128.0, nperseg=128)
    peak = max(range(len(r["coherence"])),
               key=lambda i: r["sxx"][i])
    assert r["coherence"][peak] == pytest.approx(1.0, abs=1e-6)
    assert r["n_segments"] >= 2


def test_cohere_reports_the_phase_difference():
    n = 1024
    fs = 128.0
    cyc = 16
    x = sine(n, cyc)
    y = sine(n, cyc, phase=math.pi / 2)
    r = cohere(x, y, fs=fs, nperseg=256)
    k = min(range(len(r["freqs"])),
            key=lambda i: abs(r["freqs"][i] - cyc * fs / n))
    assert abs(abs(r["phase"][k]) - math.pi / 2) < 0.3


def test_msc_is_the_square_of_the_magnitude_coherence():
    n = 1024
    x = [sine(n, 11)[i] + 0.5 * sine(n, 53)[i] for i in range(n)]
    y = [sine(n, 11)[i] + 0.5 * sine(n, 71)[i] for i in range(n)]
    r = msc(x, y, nperseg=128)
    for a, b in zip(r["msc"], r["magnitude_coherence"]):
        assert a == pytest.approx(b * b)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in r["msc"])


# ------------------------------------------------------ template matching

def test_template_finds_the_planted_copy():
    ref = [0.0, 1.0, 3.0, 1.0, 0.0]
    x = [0.0] * 20
    for i, v in enumerate(ref):
        x[8 + i] = v
    r = template(x, ref)
    assert r["best_shift"] == 8
    assert r["best_gamma"] == pytest.approx(1.0, abs=1e-9)


def test_template_normalization_beats_a_large_smooth_excursion():
    ref = [0.0, 1.0, 3.0, 1.0, 0.0]
    x = [0.0] * 30
    for i, v in enumerate(ref):
        x[5 + i] = 0.1 * v                # a faint but exact match
    for i in range(15, 25):
        x[i] = 50.0                       # a huge but shapeless plateau
    r = template(x, ref)
    assert r["best_shift"] == 5


def test_template_threshold_reports_one_peak_per_run():
    ref = [0.0, 1.0, 3.0, 1.0, 0.0]
    x = [0.0] * 40
    for start in (5, 25):
        for i, v in enumerate(ref):
            x[start + i] = v
    r = template(x, ref, threshold=0.95)
    assert r["detections"] == [5, 25]
    assert r["n_detections"] == 2


def test_template_rejects_a_degenerate_reference():
    with pytest.raises(ValueError):
        template([1.0] * 10, [2.0, 2.0, 2.0])


# ---------------------------------------------------------- matched filter

REF = [3.0, 2.0, 1.0]


def test_refpattern_eqs453_454_reverses_the_reference():
    r = refpattern()
    assert r["g"] == [3.0, 2.0, 1.0]
    assert r["h"] == [1.0, 2.0, 3.0]
    assert r["delay"] == 2
    assert r["output_is_acf"] is True
    # the book: output samples around the peak reproduce the ACF of g
    assert r["y"] == pytest.approx([3.0, 8.0, 14.0, 8.0, 3.0])


def test_mfimpulse_eq449_is_reversed_scaled_and_delayed():
    r = mfimpulse(REF)
    assert r["h"][:4] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert r["shift_samples"] == 3
    assert r["causal"] is True


def test_mfimpulse_rejects_a_shift_shorter_than_the_reference():
    with pytest.raises(ValueError):
        mfimpulse(REF, t0=1.0)


def test_mfacf_output_is_the_reference_acf():
    r = mfacf(REF)
    assert r["equals_acf"] is True
    assert r["peak_value"] == pytest.approx(r["expected_peak"])
    assert r["energy"] == pytest.approx(9.0 + 4.0 + 1.0)


def test_mftf_eq448_conjugates_the_signal_spectrum():
    X = [complex(1.0, 2.0), complex(-0.5, 0.25)]
    freqs = [0.0, 1.0]
    r = mftf(X, freqs, t0=0.0)
    assert r["H"][0] == pytest.approx(complex(1.0, -2.0))
    assert r["conjugate_of_signal"] is True


def test_mftf_delay_adds_a_linear_phase():
    X = [complex(1.0, 0.0), complex(1.0, 0.0)]
    freqs = [0.0, 0.25]
    r = mftf(X, freqs, t0=1.0)
    assert r["H"][1] == pytest.approx(
        complex(math.cos(-2 * math.pi * 0.25), math.sin(-2 * math.pi * 0.25)))


def test_mftfeeg_and_mfimpeeg_delegate_to_the_general_forms():
    X = [complex(1.0, 2.0), complex(-0.5, 0.25)]
    a = mftf(X, [0.0, 1.0], t0=0.5)["H"]
    b = mftfeeg(X, [0.0, 1.0], t0=0.5)
    assert b["H"] == pytest.approx(a)
    assert "N-1" in b["dft_shift_caveat"]
    assert mfimpeeg(REF)["h"] == pytest.approx(mfimpulse(REF)["h"])
    assert mfimpeeg(REF)["equivalent_to_correlation"] is True


def test_mfinput_eq433_scales_by_dt():
    r = mfinput([1.0, 1.0], omega=0.0, dt=0.5)
    assert r["X"].real == pytest.approx(1.0)   # 2 samples x 0.5


def test_mfoutput_eq434_peaks_where_the_filter_is_matched():
    r = mfoutput(REF, list(reversed(REF)))
    assert r["peak_index"] == 2
    assert r["peak_magnitude"] == pytest.approx(14.0)


def test_mfnoisein_eq435_halves_the_power():
    r = mfnoisein(4.0)
    assert r["density"] == pytest.approx(2.0)
    assert r["two_sided"] is True


def test_mfnoiseout_eqs436_437():
    H = [1.0, 1.0, 1.0, 1.0]
    r = mfnoiseout(4.0, H, df=0.25)
    assert r["psd"] == pytest.approx([2.0] * 4)
    assert r["power"] == pytest.approx(2.0)
    assert r["rms"] == pytest.approx(math.sqrt(2.0))


def test_mfpeak_eq438_is_a_magnitude():
    X = [complex(1.0, 0.0)] * 3
    H = [complex(1.0, 0.0)] * 3
    f = [0.0, 0.5, 1.0]
    r = mfpeak(X, H, f, t0=0.0)
    assert r["my"] == pytest.approx(1.0)


def test_mfsnr_eq439_is_peak_to_mean():
    r = mfsnr(4.0, 2.0)
    assert r["snr"] == pytest.approx(8.0)
    assert r["amplitude_snr"] == pytest.approx(4.0 / math.sqrt(2.0))
    assert r["peak_to_mean"] is True


def test_sigenergy_eq440_time_domain_is_the_integral_not_the_sum():
    # the trapezoidal rule half-weights the endpoints, so the integral of
    # x^2 is NOT the plain sum of squares -- the distinction eq. (4.40)
    # makes by writing an integral
    x = [4.0, 0.0, 0.0, 0.0]
    r = sigenergy(x, dt=1.0)
    assert r["energy"] == pytest.approx(8.0)      # 0.5 * (16 + 0)
    assert sum(v * v for v in x) == 16.0


def test_sigenergy_frequency_branch_integrates_the_spectrum():
    # a rectangular spectrum of height 2 over [0, 3] has energy 6
    X = [complex(math.sqrt(2.0), 0.0)] * 4
    freqs = [0.0, 1.0, 2.0, 3.0]
    r = sigenergy(None, X=X, freqs=freqs)
    assert r["energy_freq"] == pytest.approx(6.0)


def test_sigenergy_reports_the_gap_when_both_domains_are_given():
    x = [1.0, 1.0, 1.0, 1.0]
    X = [complex(0.0, 0.0)] * 4               # deliberately inconsistent
    r = sigenergy(x, X=X, freqs=[0.0, 1.0, 2.0, 3.0])
    assert r["parseval_holds"] is False
    assert r["max_difference"] > 0


def test_sigenergy_needs_something_to_integrate():
    with pytest.raises(ValueError):
        sigenergy(None)


def test_mfratio_eq441_reaches_its_bound_only_at_the_optimum():
    freqs = [k / 8.0 for k in range(9)]
    X = [complex(math.cos(k), math.sin(2 * k)) for k in range(9)]
    opt = mftf(X, freqs, t0=0.0)["H"]
    good = mfratio(X, opt, freqs, t0=0.0, noise_power=2.0)
    bad = mfratio(X, [complex(1.0, 0.0)] * 9, freqs, t0=0.0,
                  noise_power=2.0)
    assert good["optimality"] == pytest.approx(1.0, abs=1e-9)
    assert bad["optimality"] < 1.0
    assert good["bound"] == pytest.approx(1.0)


def test_schwarzc_eq442_equality_at_the_conjugate_condition():
    grid = [k / 8.0 for k in range(9)]
    B = [complex(math.cos(k), math.sin(k)) for k in range(9)]
    A = [3.0 * v.conjugate() for v in B]     # A = K B*
    r = schwarzc(A, B, grid)
    assert r["holds"] is True
    assert r["equality"] is True
    assert r["k"] == pytest.approx(complex(3.0, 0.0))


def test_schwarzc_is_strict_for_an_unrelated_pair():
    grid = [k / 8.0 for k in range(9)]
    A = [complex(math.cos(k), 0.0) for k in range(9)]
    B = [complex(math.sin(3 * k), 0.0) for k in range(9)]
    r = schwarzc(A, B, grid)
    assert r["holds"] is True
    assert r["equality"] is False


def test_schwarzr_eq443():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = schwarzr(a, [2.0 * v for v in a])
    assert r["equality"] is True
    assert r["k"] == pytest.approx(0.5)
    assert schwarzr(a, [5.0, 1.0, 4.0, 2.0, 3.0])["equality"] is False


def test_cauchysch_eq444_and_triangle_eq445():
    a = [3.0, 4.0]
    b = [6.0, 8.0]
    c = cauchysch(a, b)
    assert c["lhs"] == pytest.approx(50.0)
    assert c["rhs"] == pytest.approx(50.0)
    assert c["equality"] is True
    assert c["cosine"] == pytest.approx(1.0)
    t = triangle(a, b)
    assert t["lhs"] == pytest.approx(15.0)
    assert t["rhs"] == pytest.approx(15.0)
    assert t["equality"] is True
    assert triangle([1.0, 0.0], [0.0, 1.0])["equality"] is False


def test_mfpsd_eq457_output_is_real_and_nonnegative():
    r = mfpsd(REF + [0.0] * 5)
    assert r["is_psd"] is True
    assert r["max_imaginary"] < 1e-9
    assert all(v >= -1e-12 for v in r["psd"])


def test_mfmaxsnr_eq446_is_two_E_over_N0():
    r = mfmaxsnr([1.0, 1.0, 1.0, 1.0], 2.0)
    assert r["snr"] == pytest.approx(2.0 * r["energy"] / 2.0)
    assert r["depends_only_on_energy"] is True


def test_mfmaxsnr_depends_on_the_signal_only_through_its_energy():
    # two differently shaped signals with the same trapezoidal energy
    a = mfmaxsnr([0.0, 2.0, 2.0, 0.0], 1.0)          # energy 8
    b = mfmaxsnr([0.0, math.sqrt(8.0), 0.0, 0.0], 1.0)
    assert a["energy"] == pytest.approx(b["energy"])
    assert a["snr"] == pytest.approx(b["snr"])


def test_mfmaxsnr_scales_with_amplitude_squared():
    a = mfmaxsnr(sine(64, 4), 1.0)["snr"]
    b = mfmaxsnr([3.0 * v for v in sine(64, 4)], 1.0)["snr"]
    assert b == pytest.approx(9.0 * a)


def test_matchedfilt_designs_and_runs():
    ref = [1.0, 2.0, 3.0]
    x = [0.0] * 20
    for i, v in enumerate(ref):
        x[7 + i] = v
    r = matchedfilt(ref, x=x)
    assert r["whitened"] is False
    # h is the reference reversed and delayed by N = 3, so the peak lands
    # at the last sample of the planted copy plus that delay: 9 + 3 - 2
    assert r["peak_index"] == 10
    assert r["h"][:4] == pytest.approx([0.0, 3.0, 2.0, 1.0])


def test_matchedfilt_whitens_for_coloured_noise():
    ref = [1.0, 2.0, 3.0, 0.0]
    pn = [4.0, 1.0, 1.0, 1.0]
    plain = matchedfilt(ref)
    white = matchedfilt(ref, noise_psd=pn)
    assert white["whitened"] is True
    assert abs(white["H"][0]) == pytest.approx(abs(plain["H"][0]) / 4.0)


def test_matchedfilt_rejects_a_nonpositive_noise_psd():
    with pytest.raises(ValueError):
        matchedfilt([1.0, 2.0], noise_psd=[1.0, 0.0])


# ----------------------------------------------------- spectral quantities

def test_idft_eq381_inverts_the_dft():
    from morie.fn.bsaxfrm import dft
    x = [1.0, 2.0, 3.0, 4.0]
    assert idft(dft(x)["X"])["x"] == pytest.approx(x)


def test_parseval_eq391_holds_with_the_one_over_N():
    r = parseval([1.0, 2.0, 3.0, 4.0])
    assert r["energy_time"] == pytest.approx(30.0)
    assert r["holds"] is True


def test_syncsum_eq396_is_the_sum_not_the_mean():
    r = syncsum([[1.0, 2.0], [3.0, 4.0]])
    assert r["sum"] == pytest.approx([4.0, 6.0])
    assert r["average"] == pytest.approx([2.0, 3.0])
    assert r["m"] == 2


def test_specmoments_finds_a_single_tone():
    n, fs, cyc = 512, 256.0, 32
    x = sine(n, cyc)
    from morie.fn.bsastat import sigfeatures
    # build the one-sided periodogram the same way sigfeatures does
    mu = sum(x) / n
    seg = [v - mu for v in x]
    from morie.fn.bsaxfrm import dft
    X = dft(seg)["X"]
    p = [abs(X[k]) ** 2 / n for k in range(n // 2 + 1)]
    r = specmoments(p, fs=fs)
    want = cyc * fs / n
    assert r["mean_frequency"] == pytest.approx(want, abs=1.0)
    assert r["median_frequency"] == pytest.approx(want, abs=1.0)
    assert r["bandwidth"] < 3.0
    assert sigfeatures(x, fs=fs)["spectral_centroid"] == pytest.approx(
        want, abs=1.0)


def test_specmoments_flags_a_flat_spectrum():
    r = specmoments([1.0] * 65, fs=128.0)
    assert r["uniformity"] == pytest.approx(1.0)
    assert r["mean_frequency"] == pytest.approx(32.0, abs=1.0)


def test_specmoments_median_splits_the_power():
    p = [0.0] * 10 + [1.0] * 10
    r = specmoments(p, fs=40.0)
    below = sum(p[:int(r["median_frequency"] / (40.0 / 38.0))])
    assert below <= 0.5 * sum(p) + 1e-9


def test_specmoments_rejects_a_negative_psd():
    with pytest.raises(ValueError):
        specmoments([1.0, -1.0])


def test_emgfreq_mean_exceeds_median_for_a_right_skewed_spectrum():
    n, fs = 1024, 1000.0
    x = [sine(n, 40)[i] + 0.25 * sine(n, 300)[i] for i in range(n)]
    r = emgfreq(x, fs=fs)
    assert r["mean_frequency"] > r["median_frequency"]
    assert r["difference"] == pytest.approx(
        r["mean_frequency"] - r["median_frequency"])


def test_emgfreq_both_indices_fall_when_the_spectrum_shifts_down():
    n, fs = 1024, 1000.0
    fresh = emgfreq(sine(n, 200), fs=fs)
    tired = emgfreq(sine(n, 100), fs=fs)
    assert tired["mean_frequency"] < fresh["mean_frequency"]
    assert tired["median_frequency"] < fresh["median_frequency"]


def test_specres_depends_on_record_length_not_dft_length():
    a = specres(256, fs=256.0)
    b = specres(512, fs=256.0)
    assert a["delta_f"] == pytest.approx(1.0)
    assert b["delta_f"] == pytest.approx(0.5)
    assert a["zero_padding_helps"] is False


def test_specres_window_trades_resolution_for_leakage():
    rect = specres(256, fs=256.0, window="rectangular")
    black = specres(256, fs=256.0, window="blackman")
    assert black["sidelobe_db"] < rect["sidelobe_db"]
    assert black["main_lobe_bins"] > rect["main_lobe_bins"]
    assert black["resolution"] > rect["resolution"]


def test_specres_rejects_an_unknown_window():
    with pytest.raises(ValueError):
        specres(256, window="bartlett-hann-kaiser")


def test_psdhz_band_powers_carry_the_bin_width():
    p = [1.0] * 9
    r = psdhz(p, fs=16.0, bands={"low": (0.0, 4.0), "high": (4.0, 8.0)})
    assert r["bin_width"] == pytest.approx(1.0)
    assert r["band_power"]["low"] == pytest.approx(4.0)
    assert r["band_power"]["high"] == pytest.approx(4.0)
    assert r["band_fraction"]["low"] == pytest.approx(0.5)


def test_psdhz_rejects_an_inverted_band():
    with pytest.raises(ValueError):
        psdhz([1.0] * 5, fs=8.0, bands={"bad": (3.0, 1.0)})


# ----------------------------------------------------------- applications

def test_pcgsyncavg_keeps_murmur_power_that_waveform_averaging_cancels():
    n, m = 128, 12
    cycles = []
    for k in range(m):
        # a fixed S1 plus a murmur whose phase changes cycle to cycle
        cycles.append([sine(n, 3)[i]
                       + 0.8 * math.sin(2 * math.pi * 30 * i / n + k * 1.7)
                       for i in range(n)])
    r = pcgsyncavg(cycles)
    assert r["power_retained"] < 0.7      # waveform averaging loses power
    assert sum(r["average_psd"]) > sum(r["psd_of_average"])


def test_erpartifact_rejects_the_contaminated_epochs():
    good = [sine(32, 2) for _ in range(9)]
    bad = [v * 50.0 for v in sine(32, 2)]
    r = erpartifact(good + [bad], reject=5.0)
    assert r["n_rejected"] == 1
    assert r["rejected"] == [9]
    assert r["m_kept"] == 9
    assert r["snr_gain"] == pytest.approx(3.0)


def test_erpartifact_without_rejection_keeps_the_artifact():
    good = [sine(32, 2) for _ in range(9)]
    bad = [v * 50.0 for v in sine(32, 2)]
    clean = erpartifact(good + [bad], reject=5.0)["average"]
    dirty = erpartifact(good + [bad])["average"]
    assert max(abs(v) for v in dirty) > 2 * max(abs(v) for v in clean)


def test_erpartifact_refuses_when_every_epoch_is_rejected():
    with pytest.raises(ValueError):
        erpartifact([sine(32, 2) for _ in range(4)], reject=1e-6)


def test_seizcohere_tracks_bands_over_a_moving_window():
    n, fs = 2048, 128.0
    a = [sine(n, 100)[i] + 0.3 * sine(n, 13)[i] for i in range(n)]
    b = [sine(n, 100)[i] + 0.3 * sine(n, 191)[i] for i in range(n)]
    r = seizcohere([a, b], fs=fs, window=512, step=256, nperseg=128)
    assert r["n_windows"] >= 2
    assert set(r["coherence"]) == {"delta", "theta", "alpha", "beta"}
    assert all(0.0 <= v <= 1.0 + 1e-9
               for band in r["coherence"].values() for v in band)


def test_seizcohere_needs_two_channels():
    with pytest.raises(ValueError):
        seizcohere([sine(256, 5)], fs=64.0, window=128)


def test_cardioresp_plv_is_one_for_a_constant_phase_offset():
    n, fs = 512, 8.0
    resp = [math.sin(2 * math.pi * 0.25 * i / fs) for i in range(n)]
    ecg = [math.sin(2 * math.pi * 0.25 * i / fs + 0.7) for i in range(n)]
    r = cardioresp(ecg, resp, fs=fs)
    assert r["plv"] == pytest.approx(1.0, abs=0.05)


def test_cardioresp_plv_falls_for_a_drifting_phase():
    n, fs = 512, 8.0
    resp = [math.sin(2 * math.pi * 0.25 * i / fs) for i in range(n)]
    ecg = [math.sin(2 * math.pi * 0.32 * i / fs) for i in range(n)]
    locked = cardioresp(resp, resp, fs=fs)["plv"]
    drift = cardioresp(ecg, resp, fs=fs)["plv"]
    assert locked > drift


def test_cardioresp_rejects_a_band_outside_nyquist():
    with pytest.raises(ValueError):
        cardioresp(sine(64, 3), sine(64, 3), fs=8.0, band=(0.1, 9.0))


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsacorr import (rangayyan_ch3_parseval_theorem,
                                  rangayyan_ch4_dot_product_discrete,
                                  rangayyan_matched_filter_snr)
    assert rangayyan_ch4_dot_product_discrete(
        [1.0, 2.0], [3.0, 4.0])["dot_product"] == pytest.approx(11.0)
    assert rangayyan_ch3_parseval_theorem([1.0, 2.0])["holds"] is True
    assert rangayyan_matched_filter_snr([1.0, 1.0], 2.0)["snr"] > 0
