"""Rangayyan homomorphic filtering and cepstra (bsacep), Chapter 4.7.

The anchor is eq. (4.80): the complex cepstrum of a wavelet plus one
echo is the cepstrum of the wavelet plus impulses at the echo delay and
its multiples, with amplitudes (-1)^(k+1) a^k / k.  Those amplitudes are
computed here from the printed series, not read back from the code.
"""

import math

import pytest

from morie.fn.bsacep import (ccepclosed, ccepdecay, ccepstrum, ccepsum, ccepx,
                             cepstrum, convmodel, echoseries, homdeconv,
                             homofilt, hompred, lifter, logsep, mfcc, minphase,
                             multmodel, pceprel, pcepstrum, pcepsum, ratz,
                             vocaltract)

N = 64


def echo(a=0.5, n0=8, n=N):
    x = [0.0] * n
    x[0] = 1.0
    x[n0] = a
    return x


def wavelet(n=N):
    h = [0.0] * n
    h[0], h[1], h[2], h[3] = 1.0, 0.6, -0.3, 0.1
    return h


# ------------------------------------------------ the echo, eqs 4.74-4.80

def test_ccepstrum_of_an_echo_is_the_impulse_train_of_eq480():
    a, n0 = 0.5, 8
    c = ccepstrum(echo(a, n0))["cepstrum"]
    for k in (1, 2, 3, 4):
        want = ((-1) ** (k + 1)) * (a ** k) / k
        assert c[k * n0] == pytest.approx(want, abs=2e-3)


def test_ccepstrum_impulses_sit_only_at_multiples_of_the_delay():
    c = ccepstrum(echo(0.5, 8))["cepstrum"]
    off = [abs(c[i]) for i in range(1, N) if i % 8]
    on = abs(c[8])
    assert max(off) < on / 50


def test_ccepstrum_delay_is_removed_as_a_linear_phase_term():
    # a pure delay is the z^r factor of eq. (4.68)
    x = [0.0] * N
    x[5] = 1.0
    r = ccepstrum(x)
    # X(z) = z^-5, so the exponent r of eq. (4.68) is -5
    assert r["delay_removed"] == -5


def test_ccepstrum_rejects_a_vanishing_spectrum():
    # 1 - z^-1 has a zero at DC
    x = [0.0] * N
    x[0], x[1] = 1.0, -1.0
    with pytest.raises(ValueError):
        ccepstrum(x)


def test_echoseries_amplitudes_match_the_printed_series():
    r = echoseries(0.5, 8, terms=4)
    assert r["amplitudes"] == pytest.approx(
        [0.5, -0.125, 1 / 24, -0.015625])
    assert r["quefrencies"] == [8, 16, 24, 32]


def test_echoseries_matches_the_cepstrum_it_predicts():
    a, n0 = 0.4, 10
    c = ccepstrum(echo(a, n0))["cepstrum"]
    pred = echoseries(a, n0, terms=3)
    for amp, q in zip(pred["amplitudes"], pred["quefrencies"]):
        assert c[q] == pytest.approx(amp, abs=2e-3)


def test_echoseries_refuses_a_divergent_amplitude():
    with pytest.raises(ValueError):
        echoseries(1.0, 8)


def test_echoseries_converges_to_the_closed_form():
    r = echoseries(0.3, 8, terms=40, omega=0.7)
    assert r["max_error"] < 1e-12


# ------------------------------------------------- models, eqs 4.58-4.66

def test_multmodel_and_logsep_eqs458_459():
    x, p = [2.0, 3.0, 4.0], [5.0, 0.5, 2.0]
    assert multmodel(x, p)["y"] == pytest.approx([10.0, 1.5, 8.0])
    r = logsep(x, p)
    assert r["additive"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-12)


def test_logsep_enforces_the_nonzero_condition_of_eq459():
    with pytest.raises(ValueError):
        logsep([1.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        logsep([1.0, -1.0], [1.0, 1.0])


def test_convmodel_eq461():
    assert convmodel([1.0, 2.0], [3.0, 4.0])["y"] == pytest.approx(
        [3.0, 10.0, 8.0])


def test_ccepsum_eq466_residual_is_small():
    r = ccepsum(wavelet(16), echo(0.5, 4, 16))
    assert r["relative_residual"] < 0.05


# ----------------------------------------- closed form, eqs 4.67-4.73

def test_ccepclosed_eq472_minimum_phase_is_causal():
    r = ccepclosed(2.0, zeros_in=[0.5], zeros_out=[], poles_in=[0.3],
                   poles_out=[], nmax=6)
    assert r["causal"] is True
    assert r["c0"] == pytest.approx(math.log(2.0))
    # n > 0: -sum a^n/n + sum c^n/n
    assert r["positive"][0] == pytest.approx(-0.5 + 0.3)
    assert r["positive"][1] == pytest.approx(-(0.5 ** 2) / 2
                                             + (0.3 ** 2) / 2)
    assert all(abs(v) == 0 for v in r["negative"])


def test_ccepclosed_eq472_maximum_phase_is_anticausal():
    r = ccepclosed(1.0, zeros_in=[], zeros_out=[0.4], poles_in=[],
                   poles_out=[], nmax=4)
    assert r["anticausal"] is True
    assert all(abs(v) == 0 for v in r["positive"])
    assert r["negative"][-1] == pytest.approx(0.4)


def test_ccepclosed_agrees_with_the_numerical_cepstrum():
    # x(n) with a single zero at 0.5: X(z) = 1 - 0.5 z^-1
    x = [0.0] * N
    x[0], x[1] = 1.0, -0.5
    num = ccepstrum(x)["cepstrum"]
    closed = ccepclosed(1.0, zeros_in=[0.5], zeros_out=[], poles_in=[],
                        poles_out=[], nmax=5)
    for n, want in enumerate(closed["positive"], start=1):
        assert num[n] == pytest.approx(want.real, abs=1e-6)


def test_ccepclosed_reports_infinite_duration():
    assert ccepclosed(1.0, [0.5], [], [], [])["infinite_duration"] is True


def test_ratz_rejects_a_root_on_the_wrong_side():
    with pytest.raises(ValueError):
        ratz(1.0, 0, zeros_in=[1.5], zeros_out=[], poles_in=[],
             poles_out=[])


def test_ratz_evaluates_the_product_form():
    r = ratz(2.0, 0, zeros_in=[0.5], zeros_out=[], poles_in=[0.25],
             poles_out=[], z=2.0)
    assert r["X"] == pytest.approx(2.0 * (1 - 0.25) / (1 - 0.125))
    assert r["minimum_phase"] is True


def test_ccepdecay_eq473_bounds_the_numerical_cepstrum():
    x = [0.0] * N
    x[0], x[1] = 1.0, -0.5
    c = ccepstrum(x)["cepstrum"]
    b = ccepdecay([0.5], [], [], [], nmax=8)
    assert b["alpha"] == pytest.approx(0.5)
    for n, bound in enumerate(b["bound"], start=1):
        assert abs(c[n]) <= bound + 1e-9


def test_ccepdecay_flags_a_root_near_the_unit_circle():
    assert ccepdecay([0.98], [], [], [])["near_unit_circle"] is True
    assert ccepdecay([0.3], [], [], [])["near_unit_circle"] is False


# ---------------------------------------- power cepstrum, eqs 4.81-4.83

def test_pcepstrum_squaring_is_selectable():
    x = echo(0.5, 8)
    sq = pcepstrum(x, square=True)
    raw = pcepstrum(x, square=False)
    assert sq["squared"] is True
    assert raw["additivity_exact"] is True
    assert sq["cepstrum"][8] == pytest.approx(raw["cepstrum"][8] ** 2)


def test_pcepsum_eq482_is_exact_without_the_square():
    r = pcepsum(wavelet(16), echo(0.5, 4, 16), square=False)
    assert r["exact"] is True
    assert r["relative_residual"] < 1e-9


def test_pcepsum_with_the_square_leaves_a_cross_term():
    exact = pcepsum(wavelet(16), echo(0.5, 4, 16), square=False)
    squared = pcepsum(wavelet(16), echo(0.5, 4, 16), square=True)
    assert squared["relative_residual"] > exact["relative_residual"]


def test_pceprel_eq483_matches_the_direct_power_cepstrum():
    r = pceprel(echo(0.5, 8))
    assert r["relative_residual"] < 1e-9
    assert r["phase_lost"] is True


def test_real_cepstrum_is_not_invertible_but_still_shows_the_echo():
    c = cepstrum(echo(0.5, 8))
    assert c["invertible"] is False
    peak = max(range(1, N // 2), key=lambda i: abs(c["cepstrum"][i]))
    assert peak == 8


# --------------------------------------------- liftering and deconvolution

def test_lifter_is_symmetric_about_zero_quefrency():
    c = list(range(N))
    r = lifter(c, high=3, keep="low")
    kept = [i for i, v in enumerate(r["liftered"]) if v != 0.0]
    assert set(kept) <= {0, 1, 2, 3, N - 1, N - 2, N - 3}
    assert r["symmetric"] is True


def test_lifter_high_and_low_partition_the_cepstrum():
    c = [float(i + 1) for i in range(N)]
    lo = lifter(c, high=5, keep="low")["liftered"]
    hi = lifter(c, low=6, keep="high")["liftered"]
    assert [a + b for a, b in zip(lo, hi)] == pytest.approx(c)


def test_lifter_rejects_a_bad_window():
    with pytest.raises(ValueError):
        lifter([1.0, 2.0], low=5, high=1)
    with pytest.raises(ValueError):
        lifter([1.0, 2.0], keep="middle")


def test_homofilt_separates_a_slow_times_fast_product():
    n = 128
    slow = [2.0 + math.sin(2 * math.pi * i / n) for i in range(n)]
    fast = [1.0 + 0.3 * math.sin(2 * math.pi * 20 * i / n)
            for i in range(n)]
    y = [a * b for a, b in zip(slow, fast)]
    low = homofilt(y, cutoff=3, keep="low")["y"]
    # the recovered slow factor tracks the true one up to a constant
    ratio = [a / b for a, b in zip(low, slow)]
    assert max(ratio) - min(ratio) < 0.15 * (sum(ratio) / len(ratio))


def test_homofilt_needs_a_positive_signal():
    with pytest.raises(ValueError):
        homofilt([1.0, -1.0] * 8, cutoff=2)


def test_homdeconv_low_time_recovers_the_wavelet_from_an_echo():
    h = wavelet()
    y = [h[i] + 0.5 * (h[i - 12] if i >= 12 else 0.0) for i in range(N)]
    est = homdeconv(y, cutoff=6, keep="low")["y"]
    # the echo impulse at sample 12 is suppressed relative to the main
    # wavelet: in y the echo carries half the wavelet's amplitude, and
    # after low-time liftering the ratio must fall well below that
    before = max(abs(v) for v in y[12:16]) / max(abs(v) for v in y[:4])
    after = max(abs(v) for v in est[12:16]) / max(abs(v) for v in est[:4])
    assert before == pytest.approx(0.5, abs=0.05)
    assert after < before / 2


def test_hompred_components_convolve_back_to_the_signal():
    h = wavelet()
    y = [h[i] + 0.5 * (h[i - 12] if i >= 12 else 0.0) for i in range(N)]
    r = hompred(y, cutoff=6)
    assert r["relative_error"] < 1e-6
    assert len(r["low_time"]) == len(r["high_time"]) == N


def test_hompred_rejects_a_cutoff_outside_the_usable_range():
    with pytest.raises(ValueError):
        hompred(wavelet(), cutoff=0)


def test_vocaltract_finds_the_pitch_from_the_cepstral_peak():
    fs = 8000.0
    n = 512
    period = 64                        # 125 Hz
    h = [math.exp(-i / 12.0) * math.sin(2 * math.pi * 700 * i / fs)
         for i in range(48)]
    y = [0.0] * n
    for start in range(0, n - 48, period):
        for i, v in enumerate(h):
            y[start + i] += v
    r = vocaltract(y, fs=fs)
    assert r["peak_quefrency"] == pytest.approx(period, abs=4)
    # the rahmonic at 3 n0 is the larger peak; searching the whole
    # cepstrum instead of the pitch range would report it
    whole = vocaltract(y, fs=fs, pitch_range=(0.002, 0.05))
    assert whole["peak_quefrency"] > 2 * period
    assert r["pitch_hz"] == pytest.approx(fs / period, rel=0.1)


def test_minphase_preserves_the_magnitude_spectrum():
    x = [0.0] * 32
    x[0], x[5], x[11] = 1.0, -1.5, 0.4     # a mixed-phase sequence
    r = minphase(x)
    assert r["magnitude_preserved"] is True
    assert r["energy_front_loaded"] is True


# ------------------------------------------------------------------ MFCC

def test_mfcc_returns_the_requested_number_of_coefficients():
    fs = 8000.0
    x = [math.sin(2 * math.pi * 440 * i / fs) for i in range(512)]
    r = mfcc(x, fs=fs, n_filters=20, n_coeffs=13)
    assert len(r["mfcc"]) == 13
    assert len(r["filterbank_energies"]) == 20
    assert r["c0_is_energy"] is True


def test_mfcc_mel_edges_are_warped_not_linear():
    r = mfcc([0.0] * 8 + [1.0] * 8, fs=8000.0, n_filters=8, n_coeffs=4)
    e = r["edges"]
    low_gap = e[1] - e[0]
    high_gap = e[-1] - e[-2]
    assert high_gap > 1.5 * low_gap        # bands widen with frequency


def test_mfcc_c0_tracks_gain():
    fs = 8000.0
    x = [math.sin(2 * math.pi * 300 * i / fs) for i in range(256)]
    a = mfcc(x, fs=fs, n_filters=16, n_coeffs=4)["mfcc"]
    b = mfcc([4.0 * v for v in x], fs=fs, n_filters=16, n_coeffs=4)["mfcc"]
    assert b[0] > a[0]
    # the shape coefficients are unchanged by a pure gain
    assert b[1:] == pytest.approx(a[1:], abs=1e-6)


def test_mfcc_rejects_a_bad_band():
    with pytest.raises(ValueError):
        mfcc([1.0] * 32, fs=8000.0, fmin=5000.0, fmax=1000.0)


def test_ccepx_reports_the_unwrapping_diagnostics():
    r = ccepx(echo(0.5, 8))
    assert "wrapped_phase" in r
    assert r["well_conditioned"] is True
    assert r["cepstrum"] == pytest.approx(
        ccepstrum(echo(0.5, 8))["cepstrum"])


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsacep import (rangayyan_cepstrum,
                                 rangayyan_ch4_complex_cepstrum_definition,
                                 rangayyan_liftering)
    assert rangayyan_cepstrum(echo(0.5, 8))["n"] == N
    assert rangayyan_ch4_complex_cepstrum_definition(
        echo(0.5, 8))["cepstrum"][8] == pytest.approx(0.5, abs=2e-3)
    assert rangayyan_liftering([1.0] * 8, high=2)["keep"] == "low"
