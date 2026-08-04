"""Rangayyan signal-level features (bsastat): RMS, Hjorth, turns count,
SNR, synchronized averaging, fractal dimension, spectral entropy.

Expected values are hand-computed from the printed equations or are
properties the book states (a sinusoid has complexity 1; averaging M
realizations gains sqrt(M)).
"""

import math

import pytest

from morie.fn.bsastat import (fdpsd, fdvag, firingrate, formfactor, katzfd,
                              nlfeatures, obsreal, rms, sigfeatures, snr,
                              snrfilt, specentropy, syncavg, turnscount)


def sine(n, cycles, amp=1.0):
    return [amp * math.sin(2 * math.pi * cycles * i / n) for i in range(n)]


# ------------------------------------------------------------------- RMS

def test_rms_eq39_divides_by_N():
    r = rms([3.0, 4.0])
    assert r["ms"] == pytest.approx(12.5)
    assert r["rms"] == pytest.approx(math.sqrt(12.5))


def test_rms_short_time_window_is_causal():
    r = rms([1.0, 1.0, 1.0, 4.0], window=2)
    # windows: [1], [1,1], [1,1], [1,4]
    assert r["short_time"] == pytest.approx(
        [1.0, 1.0, 1.0, math.sqrt((1 + 16) / 2)])


def test_rms_rejects_a_zero_window():
    with pytest.raises(ValueError):
        rms([1.0, 2.0], window=0)


# ------------------------------------------------- Hjorth, eqs 5.25-5.26

def test_formfactor_of_a_sinusoid_tends_to_one():
    # the book states the complexity of a sinusoid is unity; the
    # difference-based derivatives approach it as sampling gets finer
    coarse = formfactor(sine(200, 5))["form_factor"]
    fine = formfactor(sine(2000, 5))["form_factor"]
    assert abs(fine - 1.0) < abs(coarse - 1.0)
    assert fine == pytest.approx(1.0, abs=2e-3)


def test_formfactor_grows_with_waveform_complexity():
    simple = formfactor(sine(2000, 5))["form_factor"]
    complexer = formfactor([a + b for a, b in
                            zip(sine(2000, 5), sine(2000, 37, 0.4))]
                           )["form_factor"]
    assert complexer > simple


def test_formfactor_is_not_the_rms_over_mean_abs_ratio():
    # the placeholder's definition; for a sinusoid it is pi/(2 sqrt 2),
    # which contradicts the book's stated value of 1
    x = sine(2000, 5)
    rms_over_mad = (math.sqrt(sum(v * v for v in x) / len(x))
                    / (sum(abs(v) for v in x) / len(x)))
    assert rms_over_mad == pytest.approx(math.pi / (2 * math.sqrt(2)),
                                         abs=1e-3)
    assert formfactor(x)["form_factor"] != pytest.approx(rms_over_mad,
                                                         abs=1e-2)


def test_formfactor_reports_activity_and_mobility():
    x = sine(1000, 3)
    r = formfactor(x)
    mu = sum(x) / len(x)
    want_activity = sum((v - mu) ** 2 for v in x) / len(x)
    assert r["activity"] == pytest.approx(want_activity)
    assert r["mobility"] > 0


def test_formfactor_refuses_a_constant_signal():
    with pytest.raises(ValueError):
        formfactor([2.0] * 10)


# -------------------------------------------------- turns count, Sec 5.6.3

def test_turnscount_counts_reversals_above_the_threshold():
    assert turnscount([0.0, 5.0, 0.0, 5.0, 0.0], threshold=1.0)["turns"] == 3


def test_turnscount_ignores_wobble_below_the_threshold():
    # a small ripple on a flat baseline: every sample is a turning point,
    # but each excursion is only 0.4, so none is a turn at threshold 1.
    # This is the distinction the book draws in Figure 5.9 between
    # turning points and Willison turns.
    x = [0.2 if i % 2 else -0.2 for i in range(40)]
    assert turnscount(x, threshold=1.0)["turns"] == 0
    assert turnscount(x, threshold=0.0)["turns"] > 30

    # a ripple riding a rise is NOT the same case: the swing from one
    # counted turn to the next is the full step, so those do count
    rise = [i + (0.6 if i % 2 else -0.6) for i in range(40)]
    assert turnscount(rise, threshold=1.0)["turns"] > 0


def test_turnscount_measures_against_the_last_counted_turn():
    # each excursion is 3, so with a threshold of 2 all reversals count,
    # and with a threshold of 4 none do
    x = [0.0, 3.0, 0.0, 3.0, 0.0, 3.0]
    assert turnscount(x, threshold=2.0)["turns"] == 4
    assert turnscount(x, threshold=4.0)["turns"] == 0


def test_turnscount_short_time_series_matches_the_window():
    r = turnscount([0.0, 5.0, 0.0, 5.0, 0.0] * 4, threshold=1.0, window=5)
    assert len(r["short_time"]) == 20
    assert max(r["short_time"]) <= 3


def test_turnscount_rejects_a_negative_threshold():
    with pytest.raises(ValueError):
        turnscount([0.0, 1.0, 0.0], threshold=-1.0)


# ------------------------------------------------------------------ SNR

def test_snr_power_and_peak_definitions_differ():
    x = sine(1000, 5)
    e = [0.1 * v for v in sine(1000, 97)]
    r = snr(x, e)
    assert r["snr_db"] == pytest.approx(r["snr_power_db"])
    # peak-to-peak of a unit sinusoid is 2, RMS noise is 0.1/sqrt2
    assert r["snr_peak_db"] == pytest.approx(
        20 * math.log10(2.0 / (0.1 / math.sqrt(2))), abs=0.2)
    assert r["snr_peak_db"] > r["snr_power_db"]


def test_snr_power_form_is_ten_log10_of_the_power_ratio():
    r = snr([1.0, -1.0, 1.0, -1.0], [0.1, -0.1, 0.1, -0.1])
    assert r["snr_db"] == pytest.approx(20.0)


def test_snr_selects_the_named_definition():
    x, e = sine(500, 3), [0.05] * 500
    assert snr(x, e, definition="peak")["snr_db"] == pytest.approx(
        snr(x, e)["snr_peak_db"])
    with pytest.raises(ValueError):
        snr(x, e, definition="whatever")


def test_snr_rejects_noiseless_input():
    with pytest.raises(ValueError):
        snr([1.0, 2.0], [0.0, 0.0])


def test_snrfilt_penalises_distortion_as_well_as_noise():
    clean = sine(400, 3)
    perfect = snrfilt(clean, clean)
    assert perfect["snr_db"] == math.inf
    # a filter that halves the signal is penalised even with no noise
    halved = snrfilt(clean, [0.5 * v for v in clean])
    assert halved["snr_db"] == pytest.approx(10 * math.log10(4.0))


def test_snrfilt_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        snrfilt([1.0, 2.0], [1.0])


# ------------------------------------------- synchronized averaging 3.95-3.96

def test_syncavg_eqs395_396_gain_is_sqrt_M():
    recs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    r = syncavg(recs)
    assert r["average"] == pytest.approx([4.0, 5.0])
    assert r["m"] == 4
    assert r["snr_gain"] == pytest.approx(2.0)
    assert r["snr_gain_db"] == pytest.approx(10 * math.log10(4.0))


def test_syncavg_shrinks_noise_by_one_over_sqrt_M():
    n, m = 64, 100
    base = sine(n, 3)
    step = 0.37
    recs = [[base[i] + 0.5 * math.sin(step * (k * n + i)) for i in range(n)]
            for k in range(m)]
    avg = syncavg(recs)["average"]
    before = max(abs(recs[0][i] - base[i]) for i in range(n))
    after = max(abs(avg[i] - base[i]) for i in range(n))
    assert after < before / 3.0


def test_syncavg_rejects_ragged_records():
    with pytest.raises(ValueError):
        syncavg([[1.0, 2.0], [1.0]])


def test_obsreal_eq395_builds_the_ensemble():
    r = obsreal([1.0, 2.0], [[0.1, 0.1], [-0.1, -0.1]])
    assert r["y"][0] == pytest.approx([1.1, 2.1])
    assert r["y"][1] == pytest.approx([0.9, 1.9])
    assert r["m"] == 2
    assert r["identical_repetitions"] is True


def test_obsreal_flags_non_identical_repetitions():
    r = obsreal([[1.0, 2.0], [1.0, 3.0]], [[0.0, 0.0], [0.0, 0.0]])
    assert r["identical_repetitions"] is False


def test_obsreal_and_syncavg_compose():
    ens = obsreal([1.0, 2.0], [[0.1, 0.1], [-0.1, -0.1]])
    assert syncavg(ens["y"])["average"] == pytest.approx([1.0, 2.0])


# ------------------------------------------ fractal dimension, eqs 6.50-6.52

def test_fdpsd_eqs650_652_on_an_exact_power_law():
    beta = 1.2
    f = [k / 10.0 for k in range(1, 200)]
    p = [v ** (-beta) for v in f]
    r = fdpsd(p, f)
    assert r["beta"] == pytest.approx(beta, abs=1e-9)
    assert r["fd"] == pytest.approx((5.0 - beta) / 2.0, abs=1e-9)
    assert r["hurst"] == pytest.approx((beta - 1.0) / 2.0, abs=1e-9)
    assert r["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert r["in_range"] is True


def test_fdpsd_flags_a_beta_outside_the_cited_range():
    f = [k / 10.0 for k in range(1, 100)]
    p = [v ** (-3.0) for v in f]           # Brownian-ish, beta = 3
    assert fdpsd(p, f)["in_range"] is False


def test_fdpsd_drops_the_dc_bin():
    f = [0.0] + [k / 10.0 for k in range(1, 100)]
    p = [1e6] + [(k / 10.0) ** (-1.0) for k in range(1, 100)]
    r = fdpsd(p, f)
    assert r["n_bins"] == 99
    assert r["beta"] == pytest.approx(1.0, abs=1e-9)


def test_fdpsd_needs_enough_bins():
    with pytest.raises(ValueError):
        fdpsd([1.0, 2.0], [1.0, 2.0])


def test_fdvag_returns_an_fd_in_the_fractal_range():
    fs = 2000.0
    n = 1024
    x = [math.sin(2 * math.pi * 150 * i / fs)
         + 0.5 * math.sin(2 * math.pi * 320 * i / fs) for i in range(n)]
    r = fdvag(x, fs=fs, fmin=100.0, fmax=500.0)
    assert 0.0 < r["fd"] < 3.0
    assert r["band"][0] >= 100.0 and r["band"][1] <= 500.0


def test_katzfd_of_a_straight_line_is_one():
    # a line has total length equal to its greatest distance, so the
    # second log term vanishes and FD = 1
    r = katzfd([float(i) for i in range(50)])
    assert r["fd"] == pytest.approx(1.0, abs=1e-9)


def test_katzfd_grows_for_a_rougher_waveform():
    smooth = katzfd(sine(500, 2))["fd"]
    rough = katzfd(sine(500, 60))["fd"]
    assert rough > smooth


def test_katzfd_records_its_scale_sensitivity():
    a = katzfd(sine(500, 5))["fd"]
    b = katzfd([10.0 * v for v in sine(500, 5)])["fd"]
    assert a != pytest.approx(b, abs=1e-6)
    assert katzfd(sine(500, 5))["scale_sensitive"] is True


# ------------------------------------------------------- spectral entropy

def test_specentropy_of_a_flat_spectrum_is_log2K():
    r = specentropy([1.0] * 8)
    assert r["entropy"] == pytest.approx(3.0)
    assert r["normalized"] == pytest.approx(1.0)


def test_specentropy_of_a_single_tone_is_zero():
    r = specentropy([0.0, 0.0, 5.0, 0.0])
    assert r["entropy"] == pytest.approx(0.0)
    assert r["normalized"] == pytest.approx(0.0)


def test_specentropy_restricts_to_a_band():
    p = [1.0, 1.0, 1.0, 1.0]
    f = [0.0, 10.0, 20.0, 30.0]
    assert specentropy(p, f, fmin=10.0, fmax=20.0)["n_bins"] == 2


def test_specentropy_rejects_a_negative_psd():
    with pytest.raises(ValueError):
        specentropy([1.0, -1.0])


# ----------------------------------------------------------- firing rate

def test_firingrate_is_the_reciprocal_of_the_mean_interval():
    t = [0.0, 0.1, 0.2, 0.3]
    r = firingrate(t)
    assert r["mean_idi"] == pytest.approx(0.1)
    assert r["mfr"] == pytest.approx(10.0)
    assert r["cv_idi"] == pytest.approx(0.0, abs=1e-15)


def test_firingrate_differs_from_the_mean_of_reciprocals():
    t = [0.0, 0.05, 0.35]            # intervals 0.05 and 0.30
    r = firingrate(t)
    assert r["mfr"] == pytest.approx(1.0 / 0.175)
    assert r["mean_instantaneous_rate"] == pytest.approx(
        (1 / 0.05 + 1 / 0.30) / 2)
    assert r["mfr"] < r["mean_instantaneous_rate"]


def test_firingrate_converts_sample_indices():
    r = firingrate([0, 100, 200], fs=1000.0)
    assert r["mfr"] == pytest.approx(10.0)


def test_firingrate_rejects_unsorted_instants():
    with pytest.raises(ValueError):
        firingrate([0.0, 0.2, 0.1])


# --------------------------------------------------------- feature vectors

def test_sigfeatures_agrees_with_the_individual_measures():
    x = sine(512, 7)
    r = sigfeatures(x, fs=256.0)
    assert r["rms"] == pytest.approx(rms(x)["rms"])
    assert r["form_factor"] == pytest.approx(formfactor(x)["form_factor"])
    assert r["turns"] == turnscount(x, threshold=0.0)["turns"]


def test_sigfeatures_centroid_finds_the_tone():
    fs, n, f0 = 256.0, 512, 16.0
    x = [math.sin(2 * math.pi * f0 * i / fs) for i in range(n)]
    assert sigfeatures(x, fs=fs)["spectral_centroid"] == pytest.approx(
        f0, abs=0.5)


def test_nlfeatures_returns_a_slot_per_component():
    r = nlfeatures(sine(256, 5))
    assert set(r["features"]) == {"apen", "sampen", "dfa", "lyapunov"}
    # any component that could not be computed must say why
    for key, val in r["features"].items():
        assert (val is not None) or (key in r["failures"])


def test_nlfeatures_needs_a_usable_record():
    with pytest.raises(ValueError):
        nlfeatures([1.0, 2.0, 3.0])


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsastat import (rangayyan_form_factor, rangayyan_rms,
                                  rangayyan_turns_count)
    assert rangayyan_rms([3.0, 4.0])["rms"] == pytest.approx(math.sqrt(12.5))
    assert rangayyan_turns_count([0.0, 5.0, 0.0], threshold=1.0)["turns"] == 1
    assert rangayyan_form_factor(sine(2000, 5))["form_factor"] == \
        pytest.approx(1.0, abs=2e-3)
