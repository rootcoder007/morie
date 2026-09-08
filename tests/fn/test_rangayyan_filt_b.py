"""Rangayyan filter design (bsafilt A2 + B + C): the 8-point moving
average, the integrator, the difference operators, the baseline-wander
filter, the Butterworth family, the bilinear transformation, the
DFT-indexed responses, the notch and comb filters, the windowed sinc and
the window functions.

The book's own worked fourth-order example, eqs. (3.147)-(3.148), is used
as the oracle: the pole coordinates and gain asserted below are numbers
the book prints, not this implementation's output.
"""

import math

import pytest

from morie.fn.bsafilt import (bilinear, bilinunit, bilinunwarp, bilinwarp,
                              blackman, bwanalog, bwander, bwandereq,
                              bwanderz, bwdigital, bwdirect, bwhp,
                              bwhpdft, bwlp, bwlpdft, bwpoles, bwsqlap,
                              bwsqmag, cdiff3, cdiff3mag, cdiff3ph,
                              cdiff3tf, comb, diff1, diff2, fdiff,
                              fdifffr, fdiffmag, fdiffph, fdifftf,
                              freqresp, grpdelay, hamming, hannwin,
                              intfr, intft, intmag, intph, iirdiffgen,
                              ma8fr, ma8imp, ma8rec, ma8rectf, ma8sinc,
                              ma8tf, mfilth, notch, notch60, phaseresp,
                              runint, runintall, sinckern, windowfn)

# eqs (3.147)-(3.148): the book's fourth-order Butterworth example.
# Its poles are printed as (-0.556072 +- j1.342475) and
# (-1.342475 +- j0.556072); the squared pole radius is 2.111456 and the
# gain is 4.458247.
BOOK_WC = math.sqrt(2.111456)


# --------------------------------------------------- the 8-point average

def test_ma8imp_eq3109_is_eight_equal_taps():
    r = ma8imp()
    assert r["h"] == pytest.approx([0.125] * 8)
    assert r["sum"] == pytest.approx(1.0)
    assert r["equal_weights"] is True
    assert ma8imp(n=9)["value"] == 0.0


def test_ma8tf_eq3110_has_seven_zeros_and_unit_dc_gain():
    r = ma8tf(1.0)
    assert r["H"].real == pytest.approx(1.0)
    assert r["n_zeros"] == 7
    assert r["dc_gain"] == pytest.approx(1.0)


def test_ma8tf_vanishes_at_every_multiple_of_fs_over_eight():
    # zeros at w = 2 pi k / 8 for k = 1..7
    for k in range(1, 8):
        w = 2.0 * math.pi * k / 8.0
        z = complex(math.cos(w), math.sin(w))
        assert abs(ma8tf(z)["H"]) == pytest.approx(0.0, abs=1e-12)


def test_ma8fr_eq3111_factored_form_is_exact():
    r = ma8fr([0.0, 0.3, 1.0, 2.0, math.pi])
    assert r["factored_form_agrees"] is True
    assert r["max_difference"] < 1e-12
    assert r["bracket_is_inside_the_product"] is True


def test_ma8fr_the_docstrings_product_form_would_have_been_wrong():
    # {1 + exp(-j4w)}{1 + 2cos w + 2cos 2w + 2cos 3w} is NOT the book's
    # form and does not equal the sum
    w = 1.0
    brack = 1.0 + 2 * math.cos(w) + 2 * math.cos(2 * w) + 2 * math.cos(3 * w)
    wrong = 0.125 * (1.0 + complex(math.cos(-4 * w), math.sin(-4 * w))) * brack
    assert abs(wrong - ma8fr(w)["H"]) > 1e-3


def test_ma8rec_eq3120_matches_the_direct_form():
    x = [1, 4, 2, 8, 5, 7, 3, 9, 6, 2, 4, 1, 8]
    r = ma8rec(x)
    assert r["agrees_with_direct_form"] is True
    assert r["additions_per_sample"] == 2
    assert r["error_accumulates"] is True


def test_ma8rectf_eq3121_is_one_at_the_removable_singularity():
    assert ma8rectf(1.0)["H"].real == pytest.approx(1.0)
    assert ma8rectf(1.0)["removable_singularity_at_z_equals_one"] is True
    assert ma8rectf(1.0)["still_fir"] is True
    with pytest.raises(ValueError):
        ma8rectf(0.0)


def test_ma8sinc_eq3122_agrees_with_eq3111_and_delays_by_seven_halves():
    r = ma8sinc([0.0, 0.3, 1.0, 2.0])
    assert r["agrees_with_eq_3_111"] is True
    assert r["group_delay"] == pytest.approx(3.5)
    assert r["delay_is_not_an_integer"] is True


def test_the_measured_group_delay_of_the_8_point_average_is_seven_halves():
    g = grpdelay([0.125] * 8, fs=1000.0, n_freqs=257)
    assert g["mean"] == pytest.approx(3.5, abs=1e-9)


# --------------------------------------------------------- the integrator

def test_runint_eq3112_over_a_unit_signal_is_the_window_length():
    t = [i / 100.0 for i in range(201)]      # 0 .. 2
    x = [1.0] * 201
    r = runint(x, t, 0.5)
    assert r["y"][-1] == pytest.approx(0.5, abs=1e-9)
    assert r["clipped_windows"] > 0          # the leading windows are short
    with pytest.raises(ValueError):
        runint(x, t, 0.0)


def test_runintall_eq3113_is_the_cumulative_integral():
    t = [i / 100.0 for i in range(101)]
    r = runintall([1.0] * 101, t)
    assert r["y"][0] == 0.0
    assert r["total"] == pytest.approx(1.0, abs=1e-12)
    assert r["constant_of_integration_is_arbitrary"] is True
    assert r["discrete_pole_on_the_unit_circle"] is True


def test_intfr_eq3116_and_its_magnitude_and_phase():
    assert intfr(2.0)["H"] == pytest.approx(complex(0.0, -0.5))
    assert intmag(2.0)["magnitude"] == pytest.approx(0.5)
    assert intph(2.0)["phase"] == pytest.approx(-math.pi / 2.0)
    # the magnitude must be positive for negative omega too
    assert intmag(-2.0)["magnitude"] == pytest.approx(0.5)
    assert intph(-2.0)["phase"] == pytest.approx(math.pi / 2.0)


def test_the_integrator_is_unbounded_at_dc():
    for f in (intfr, intmag, intph):
        with pytest.raises(ValueError):
            f(0.0)


def test_intft_eq3115_returns_the_delta_weight_separately():
    r = intft(2.0, 1.0, X0=3.0)
    assert r["Y"] == pytest.approx(complex(0.0, -2.0))
    assert r["delta_weight"] == pytest.approx(math.pi * 3.0)
    assert r["at_dc"] is False
    z = intft(2.0, 0.0, X0=3.0)
    assert z["Y"] is None                    # 1/(jw) is unbounded there
    assert z["at_dc"] is True


# ------------------------------------------------- the difference operators

def test_fdiff_eq3123_scales_by_the_sampling_interval():
    a = fdiff([0.0, 1.0, 3.0], T=1.0)["y"]
    b = fdiff([0.0, 1.0, 3.0], T=0.5)["y"]
    assert a == pytest.approx([0.0, 1.0, 2.0])
    assert b == pytest.approx([0.0, 2.0, 4.0])
    assert fdiff([1.0], T=1.0)["highpass"] is True
    with pytest.raises(ValueError):
        fdiff([1.0, 2.0], T=0.0)


def test_fdifftf_eq3124_has_its_only_zero_at_dc():
    r = fdifftf(1.0)
    assert abs(r["H"]) == pytest.approx(0.0, abs=1e-15)
    assert r["zeros"] == [1.0]
    assert r["dc_gain"] == 0.0


def test_fdifffr_eq3125_both_forms_agree():
    r = fdifffr([0.0, 0.5, 1.5, math.pi])
    assert r["forms_agree"] is True
    assert r["half_sample_delay"] == pytest.approx(0.5)


def test_fdiffmag_eq3126_and_phase_eq3127():
    assert fdiffmag(math.pi)["magnitude"] == pytest.approx(2.0)
    assert fdiffmag(0.0)["magnitude"] == pytest.approx(0.0)
    assert fdiffph(1.0)["phase"] == pytest.approx(math.pi / 2.0 - 0.5)
    assert fdiffph(1.0)["group_delay"] == pytest.approx(0.5)
    # and they agree with the complex response
    for w in (0.4, 1.2, 2.5):
        H = fdifffr(w)["H"]
        assert abs(H) == pytest.approx(fdiffmag(w)["magnitude"], abs=1e-12)
        assert math.atan2(H.imag, H.real) == pytest.approx(
            fdiffph(w)["phase"], abs=1e-12)


def test_cdiff3_eq3128_is_the_mean_of_two_first_differences():
    r = cdiff3([1, 4, 2, 8, 5, 7])
    assert r["derivation_agrees"] is True
    assert r["controls_noise_amplification"] is True
    assert r["poor_above_fs_over_10"] is True


def test_cdiff3tf_eq3129_factors_into_a_difference_and_a_two_point_mean():
    r = cdiff3tf([0.8, 1.3, complex(0.0, 1.0)])
    assert r["cascade_agrees"] is True
    assert r["zeros"] == [1.0, -1.0]
    assert r["bandpass"] is True


def test_cdiff3mag_eq3130_vanishes_at_both_ends():
    assert cdiff3mag(0.0)["magnitude"] == pytest.approx(0.0)
    assert cdiff3mag(math.pi)["magnitude"] == pytest.approx(0.0, abs=1e-15)
    assert cdiff3mag(math.pi / 2.0)["magnitude"] == pytest.approx(1.0)


def test_cdiff3ph_eq3131_has_a_whole_sample_group_delay():
    assert cdiff3ph(1.0)["phase"] == pytest.approx(math.pi / 2.0 - 1.0)
    assert cdiff3ph(1.0)["group_delay"] == pytest.approx(1.0)
    # a whole sample, against the half sample of the plain difference
    assert cdiff3ph(1.0)["group_delay"] > fdiffph(1.0)["group_delay"]


def test_diff2_is_two_first_differences_in_cascade():
    r = diff2([1, 4, 2, 8, 5])
    assert r["cascade_agrees"] is True
    assert r["zeros"] == [1.0, 1.0]
    assert r["gain_rises_quadratically"] is True


def test_diff1_reports_the_coefficients_of_the_operator():
    r = diff1([1.0, 3.0, 6.0])
    assert r["y"] == pytest.approx([1.0, 2.0, 3.0])
    assert r["b"] == pytest.approx([1.0, -1.0])
    assert r["highpass"] is True


# ------------------------------------------------ the baseline-wander filter

def test_bwander_eq3132_kills_dc_and_passes_the_rest():
    r = bwander(1.0)
    assert abs(r["H"]) == pytest.approx(0.0, abs=1e-15)
    assert r["poles"] == [0.995]
    assert r["no_longer_fir"] is True
    # well away from DC the gain is close to unity
    w = math.pi / 2.0
    assert abs(bwander(complex(math.cos(w), math.sin(w)))["H"]) == \
        pytest.approx(1.0, abs=0.01)


def test_bwander_refuses_a_pole_on_the_unit_circle():
    with pytest.raises(ValueError):
        bwander(0.5, pole=1.0)
    with pytest.raises(ValueError):
        bwander(0.0)


def test_bwanderz_eq3133_is_the_same_filter():
    r = bwanderz([0.8, 1.3, complex(0.2, 0.9)])
    assert r["forms_agree"] is True
    assert r["numerator_is_the_distance_to_the_zero"] is True


def test_bwandereq_eq3134_carries_a_plus_on_the_feedback():
    r = bwandereq([1.0, 0.0, 0.0, 0.0])
    assert r["feedback_sign"] == "+"
    # a decaying tail, not an alternating one -- the pole is at +0.995
    assert r["y"][1] < 0 and r["y"][2] < 0
    assert abs(r["y"][2]) < abs(r["y"][1])
    assert r["iir"] is True


def test_bwandereq_removes_a_constant_offset():
    # the step at the start decays as pole^n, so the time constant is
    # 1/(1 - 0.995) = 200 samples; after 50 it is nowhere near settled
    assert abs(bwandereq([5.0] * 50)["y"][-1]) > 3.0
    settled = bwandereq([5.0] * 4000)["y"]
    assert abs(settled[-1]) < 1e-6


# ---------------------------------------------------- the Butterworth family

def test_bwsqmag_eq3135_is_half_power_at_cutoff_for_every_order():
    for n in (1, 2, 4, 8):
        assert bwsqmag(2.0, 2.0, n)["squared_magnitude"] == pytest.approx(0.5)
    assert bwsqmag(0.0, 2.0, 4)["squared_magnitude"] == pytest.approx(1.0)
    assert bwsqmag([1.0, 2.0, 4.0], 2.0, 4)["monotonic"] is True


def test_bwsqmag_is_monotonically_decreasing():
    v = bwsqmag([0.0, 1.0, 2.0, 4.0, 8.0], 2.0, 4)["squared_magnitude"]
    assert all(b <= a + 1e-15 for a, b in zip(v, v[1:]))
    with pytest.raises(ValueError):
        bwsqmag(1.0, 0.0, 4)
    with pytest.raises(ValueError):
        bwsqmag(1.0, 2.0, 0)


def test_bwsqlap_eq3136_has_2n_poles_half_unusable():
    r = bwsqlap(complex(0.3, 0.4), 2.0, 3)
    assert r["n_poles"] == 6
    assert r["half_are_right_half_plane"] is True
    assert r["not_a_filter_until_the_poles_are_selected"] is True


def test_bwpoles_eq3137_reproduces_the_books_worked_example():
    r = bwpoles(BOOK_WC, 4)
    got = sorted((round(p.real, 6), round(abs(p.imag), 6))
                 for p in r["left_half_plane"])
    assert got == [(-1.342475, 0.556072), (-1.342475, 0.556072),
                   (-0.556072, 1.342475), (-0.556072, 1.342475)]
    assert r["n_left_half_plane"] == 4
    assert r["none_on_the_imaginary_axis"] is True


def test_bwpoles_lie_on_a_circle_spaced_evenly():
    n = 5
    r = bwpoles(3.0, n)
    assert all(abs(abs(p) - 3.0) < 1e-12 for p in r["poles"])
    assert r["angular_spacing"] == pytest.approx(math.pi / n)
    assert r["real_pole_for_odd_order"] is True
    assert r["value"] is None
    assert bwpoles(3.0, n, k=1)["value"] == r["poles"][0]
    with pytest.raises(ValueError):
        bwpoles(3.0, n, k=0)


def test_bwanalog_eq3138_reproduces_the_books_gain():
    r = bwanalog(BOOK_WC, 4)
    assert r["gain"] == pytest.approx(4.458247, abs=1e-5)
    assert r["coefficients_are_real"] is True
    assert r["left_half_plane_only"] is True
    # and its DC value is unity by construction
    assert abs(bwanalog(BOOK_WC, 4, s=0.0)["H"]) == pytest.approx(1.0)


def test_bwanalog_is_half_power_at_the_cutoff():
    Wc = 2.0
    H = bwanalog(Wc, 4, s=complex(0.0, Wc))["H"]
    assert abs(H) == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-9)


# ------------------------------------------------ the bilinear transformation

def test_bilinear_eq3139_maps_dc_and_refuses_the_point_at_infinity():
    assert bilinear(1.0)["s"] == pytest.approx(complex(0.0, 0.0))
    assert bilinear(1.0)["stability_is_preserved"] is True
    with pytest.raises(ValueError):
        bilinear(-1.0)
    with pytest.raises(ValueError):
        bilinear(0.0)


def test_bilinear_maps_the_left_half_plane_inside_the_unit_disc():
    # a stable analog pole must come back inside the circle
    for p in (complex(-0.5, 1.0), complex(-2.0, 0.0), complex(-0.1, 3.0)):
        z = (2.0 + p) / (2.0 - p)            # T = 1
        assert abs(z) < 1.0


def test_bilinunit_eq3140_has_no_real_part():
    r = bilinunit([0.2, 1.0, 2.5])
    assert r["sigma_vanishes"] is True
    assert r["forms_agree"] is True


def test_bilinwarp_eq3141_and_its_inverse_eq3142():
    assert bilinwarp(0.0)["Omega"] == pytest.approx(0.0)
    assert bilinunwarp(0.0)["omega"] == pytest.approx(0.0)
    r = bilinunwarp([0.5, 2.0, 10.0, 1000.0])
    assert r["inverts_eq_3_141"] is True
    assert r["always_inside_the_open_interval"] is True
    with pytest.raises(ValueError):
        bilinwarp(math.pi)


def test_the_warping_is_nonlinear():
    # doubling the digital frequency does not double the analog one
    a = bilinwarp(0.5)["Omega"]
    b = bilinwarp(1.0)["Omega"]
    assert b != pytest.approx(2.0 * a, rel=1e-3)


# ------------------------------------------------------ the digital designs

def test_bwdigital_eq3143_puts_every_zero_at_minus_one():
    r = bwdigital(N=4, fc=100.0, fs=1000.0)
    assert r["zeros_at_minus_one"] == 4
    assert r["zeros_are_forced_by_the_bilinear_transform"] is True
    assert r["leading_a_is_one"] is True
    assert sum(r["b"]) / sum(r["a"]) == pytest.approx(1.0, abs=1e-12)


def test_bwdigital_needs_exactly_one_cutoff_specification():
    with pytest.raises(ValueError):
        bwdigital(N=4)
    with pytest.raises(ValueError):
        bwdigital(N=4, Omega_c=1.0, fc=100.0, fs=1000.0)
    with pytest.raises(ValueError):
        bwdigital(N=4, fc=100.0)
    with pytest.raises(ValueError):
        bwdigital(N=4, fc=600.0, fs=1000.0)


def test_bwlp_is_half_power_at_the_requested_cutoff():
    lp = bwlp(100.0, order=4, fs=1000.0)
    w = 2.0 * math.pi * 100.0 / 1000.0
    z = complex(math.cos(w), math.sin(w))
    H = bwlp(100.0, order=4, fs=1000.0, z=z)["H"]
    assert abs(H) == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-9)
    assert lp["prewarped"] is True
    assert lp["kind"] == "lowpass"


def test_bwhp_is_half_power_at_the_cutoff_and_zero_at_dc():
    w = 2.0 * math.pi * 100.0 / 1000.0
    z = complex(math.cos(w), math.sin(w))
    hp = bwhp(100.0, order=4, fs=1000.0, z=z)
    assert abs(hp["H"]) == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-9)
    assert abs(bwhp(100.0, order=4, fs=1000.0, z=1.0)["H"]) == \
        pytest.approx(0.0, abs=1e-12)
    assert hp["normalized_at_nyquist"] is True


def test_a_higher_order_gives_a_sharper_transition():
    def gain(order, f):
        w = 2.0 * math.pi * f / 1000.0
        z = complex(math.cos(w), math.sin(w))
        return abs(bwlp(100.0, order=order, fs=1000.0, z=z)["H"])
    assert gain(8, 200.0) < gain(2, 200.0)
    assert gain(8, 100.0) == pytest.approx(gain(2, 100.0), abs=1e-6)


def test_iirdiffgen_eq3144_runs_a_designed_filter():
    lp = bwlp(100.0, order=2, fs=1000.0)
    y = iirdiffgen([1.0] * 80, lp["b"], lp["a"][1:])["y"]
    assert y[-1] == pytest.approx(1.0, abs=1e-6)      # unit DC gain


# ------------------------------------------------- the DFT-indexed responses

def test_bwdirect_eq3145_is_half_power_at_cutoff_and_zero_phase():
    r = bwdirect(1.0, 1.0, 4)
    assert r["squared_magnitude"] == pytest.approx(0.5)
    assert r["zero_phase"] is True
    assert r["not_causal"] is True
    assert r["no_warping"] is True


def test_bwlpdft_eq3146_reflects_the_upper_half():
    r = bwlpdft(16, kc=4, N=2)
    m = r["magnitude"]
    assert len(m) == 16
    assert all(m[k] == pytest.approx(m[16 - k]) for k in range(1, 8))
    assert m[0] == pytest.approx(1.0)
    assert r["squared_magnitude"][4] == pytest.approx(0.5)


def test_bwlpdft_uses_a_ceiling_for_the_cutoff_index():
    # K wc/ws = 100 * 33/1000 = 3.3, so kc must be 4, not 3
    r = bwlpdft(100, fc=33.0, fs=1000.0, N=2)
    assert r["kc"] == 4
    assert r["cutoff_index_uses_a_ceiling"] is True


def test_bwhpdft_eq3149_is_exactly_zero_at_dc():
    r = bwhpdft(16, kc=4, N=2)
    assert r["magnitude"][0] == 0.0
    assert r["squared_magnitude"][4] == pytest.approx(0.5)
    assert r["leaves_high_frequency_noise_untouched"] is True


def test_the_dft_lowpass_and_highpass_are_complementary_in_power():
    lo = bwlpdft(32, kc=8, N=3)["squared_magnitude"]
    hi = bwhpdft(32, kc=8, N=3)["squared_magnitude"]
    for k in range(1, 17):
        assert lo[k] + hi[k] == pytest.approx(1.0, abs=1e-12)


def test_the_dft_designs_refuse_a_bad_specification():
    with pytest.raises(ValueError):
        bwlpdft(16)
    with pytest.raises(ValueError):
        bwlpdft(16, kc=4, fc=10.0, fs=100.0)
    with pytest.raises(ValueError):
        bwhpdft(16, fc=60.0, fs=100.0)


# --------------------------------------------------- notch, comb, sinc, windows

def test_notch60_puts_a_conjugate_pair_of_zeros_on_the_interference():
    r = notch60(1000.0, 60.0)
    assert r["gain_at_the_notch"] == pytest.approx(0.0, abs=1e-12)
    assert r["dc_gain"] == pytest.approx(1.0)
    assert r["fir"] is True
    assert r["notch_is_wide_without_poles"] is True
    with pytest.raises(ValueError):
        notch60(1000.0, 600.0)


def bandwidth_3db(gain_at, f0, fs):
    """Width of the band about f0 where the gain is below 1/sqrt(2)."""
    lo, hi = f0, f0
    step = 0.05
    while lo > step and gain_at(lo) < 1.0 / math.sqrt(2.0):
        lo -= step
    while hi < fs / 2.0 - step and gain_at(hi) < 1.0 / math.sqrt(2.0):
        hi += step
    return hi - lo


def test_notch_with_poles_is_narrower_than_zeros_alone():
    fs, f0 = 1000.0, 60.0
    wide = notch60(fs, f0)

    def g_wide(f):
        w = 2.0 * math.pi * f / fs
        z = complex(math.cos(w), math.sin(w))
        return abs(sum(wide["b"][k] * z ** -k
                       for k in range(len(wide["b"]))))

    def g_narrow(f):
        w = 2.0 * math.pi * f / fs
        z = complex(math.cos(w), math.sin(w))
        return abs(notch(f0, r=0.98, fs=fs, z=z)["H"])

    # the poles pull the -3 dB width in several-fold
    bw_narrow = bandwidth_3db(g_narrow, f0, fs)
    bw_wide = bandwidth_3db(g_wide, f0, fs)
    assert bw_narrow < bw_wide / 5.0
    # and the measured width matches the documented bw = (1 - r) fs / pi
    assert bw_narrow == pytest.approx((1.0 - 0.98) * fs / math.pi, abs=0.1)
    narrow = notch(f0, r=0.98, fs=fs)
    assert narrow["poles_narrow_the_notch"] is True
    assert narrow["gain_at_the_notch"] == pytest.approx(0.0, abs=1e-12)


def test_notch_needs_exactly_one_of_bandwidth_or_radius():
    with pytest.raises(ValueError):
        notch(60.0, fs=1000.0)
    with pytest.raises(ValueError):
        notch(60.0, bandwidth=4.0, r=0.98, fs=1000.0)
    with pytest.raises(ValueError):
        notch(60.0, r=1.0, fs=1000.0)


def test_a_narrower_bandwidth_needs_a_pole_closer_to_the_circle():
    assert notch(60.0, bandwidth=2.0, fs=1000.0)["r"] > \
        notch(60.0, bandwidth=8.0, fs=1000.0)["r"]


def test_comb_notches_every_harmonic_at_once():
    r = comb(20, fs=1000.0)
    assert r["notch_spacing_hz"] == pytest.approx(50.0)
    assert r["notch_frequencies_hz"][:3] == pytest.approx([0.0, 50.0, 100.0])
    assert r["dc_gain"] == 0.0
    assert r["removes_dc_as_well"] is True
    for f in (50.0, 100.0, 150.0):
        w = 2.0 * math.pi * f / 1000.0
        z = complex(math.cos(w), math.sin(w))
        assert abs(comb(20, fs=1000.0, z=z)["H"]) == pytest.approx(
            0.0, abs=1e-12)


def test_sinckern_is_symmetric_and_normalized():
    r = sinckern(100.0, fs=1000.0, M=32)
    assert sum(r["h"]) == pytest.approx(1.0, abs=1e-12)
    assert r["delay_samples"] == pytest.approx(16.0)
    assert r["h"][:16] == pytest.approx(list(reversed(r["h"][17:])))
    assert r["truncation_causes_gibbs_ripple"] is True


def test_a_window_removes_the_gibbs_ripple_flag_and_lowers_the_sidelobes():
    plain = sinckern(100.0, fs=1000.0, M=64)
    tapered = sinckern(100.0, fs=1000.0, M=64, window="hamming")
    assert tapered["truncation_causes_gibbs_ripple"] is False
    a = freqresp(plain["h"], fs=1000.0, n_freqs=1024)["magnitude"]
    b = freqresp(tapered["h"], fs=1000.0, n_freqs=1024)["magnitude"]
    stop = slice(400, 1024)                  # well inside the stopband
    assert max(b[stop]) < max(a[stop])


def test_the_windows_have_the_book_coefficients():
    assert hannwin(9)["endpoints"] == pytest.approx([0.0, 0.0])
    assert hannwin(9)["reaches_zero_at_the_ends"] is True
    assert hamming(9)["endpoints"] == pytest.approx([0.08, 0.08])
    assert hamming(9)["reaches_zero_at_the_ends"] is False
    assert blackman(9)["endpoints"] == pytest.approx([0.0, 0.0], abs=1e-15)
    for w in (hannwin(15), hamming(15), blackman(15)):
        assert w["symmetric"] is True
        assert max(w["w"]) == pytest.approx(1.0, abs=1e-9)


def test_the_hann_window_is_not_the_hann_filter():
    assert hannwin(9)["not_the_hann_filter_of_eq_3_100"] is True
    assert hannwin(3)["w"] != pytest.approx([0.25, 0.5, 0.25])


def test_windowfn_dispatches_and_names_the_rectangular_default():
    assert windowfn(5, "rectangular")["w"] == pytest.approx([1.0] * 5)
    assert windowfn(9, "hann")["w"] == pytest.approx(hannwin(9)["w"])
    assert windowfn(9, "hamming")["w"] == pytest.approx(hamming(9)["w"])
    assert windowfn(9, "blackman")["w"] == pytest.approx(blackman(9)["w"])
    assert windowfn(5, "hann")["doing_nothing_is_the_rectangular_window"] \
        is True
    with pytest.raises(ValueError):
        windowfn(5, "bartlett")
    with pytest.raises(ValueError):
        windowfn(0)


def test_a_single_point_window_is_unity():
    for f in (hannwin, hamming, blackman):
        assert f(1)["w"] == [1.0]


# ------------------------------------------------ responses and group delay

def test_freqresp_runs_dc_to_nyquist_inclusive():
    r = freqresp([0.25, 0.5, 0.25], fs=1000.0, n_freqs=101)
    assert r["f"][0] == 0.0
    assert r["f"][-1] == pytest.approx(500.0)
    assert r["magnitude"][0] == pytest.approx(1.0)
    assert r["magnitude"][-1] == pytest.approx(0.0, abs=1e-15)
    assert r["one_sided"] is True


def test_freqresp_refuses_a_pole_on_the_unit_circle():
    with pytest.raises(ValueError):
        freqresp([1.0], a=[1.0, -1.0], fs=1000.0, n_freqs=8)
    with pytest.raises(ValueError):
        freqresp([1.0], a=[0.0, 1.0])


def test_phaseresp_unwraps_by_default():
    r = phaseresp([0.25, 0.5, 0.25], fs=1000.0, n_freqs=101)
    assert r["unwrap"] is True
    # the Hann filter's phase is -omega, monotone once unwrapped.  Nyquist
    # is excluded: the response is exactly zero there and has no phase.
    good = [v for v, ok in zip(r["unwrapped"], r["defined"]) if ok]
    assert all(b <= a + 1e-12 for a, b in zip(good, good[1:]))
    assert r["n_undefined"] == 1
    assert r["phase_undefined_where_the_response_vanishes"] is True
    assert r["wrapping_is_an_arctangent_artifact"] is True
    # and the slope is -1 in omega, matching eq. (3.107)
    w = [2.0 * math.pi * f / 1000.0 for f, ok in zip(r["f"], r["defined"])
         if ok]
    assert (good[-1] - good[0]) / (w[-1] - w[0]) == pytest.approx(
        -1.0, abs=1e-9)


def test_group_delay_of_a_symmetric_fir_is_half_its_order():
    for taps, want in (([0.25, 0.5, 0.25], 1.0),
                       ([1 / 3.0] * 3, 1.0),
                       ([0.125] * 8, 3.5)):
        g = grpdelay(taps, fs=1000.0, n_freqs=257)
        assert g["mean"] == pytest.approx(want, abs=1e-9)


def test_group_delay_survives_a_zero_on_the_unit_circle():
    # the three-point mean has a zero at w = 2 pi / 3; differentiating the
    # phase there gives a spike, the coefficient formula does not
    g = grpdelay([1 / 3.0] * 3, fs=1000.0, n_freqs=257)
    assert g["mean"] == pytest.approx(1.0, abs=1e-9)
    assert g["from_the_coefficients"] is True
    assert g["phase_differentiation_breaks_at_unit_circle_zeros"] is True


def test_an_iir_filter_has_a_frequency_dependent_group_delay():
    lp = bwlp(100.0, order=4, fs=1000.0)
    assert grpdelay(lp["b"], lp["a"], fs=1000.0)["approximately_constant"] \
        is False


# ---------------------------------------------------------- matched filter

def test_mfilth_reverses_the_template():
    r = mfilth([1, 2, 3])
    assert r["h"] == pytest.approx([3.0, 2.0, 1.0])
    assert r["peak_index"] == 2
    assert r["time_reversed"] is True
    assert r["energy"] == pytest.approx(14.0)


def test_mfilth_normalization_gives_unit_energy():
    r = mfilth([1, 2, 3], normalize=True)
    assert sum(v * v for v in r["h"]) == pytest.approx(1.0)
    assert r["normalized"] is True
    with pytest.raises(ValueError):
        mfilth([0.0, 0.0], normalize=True)
    with pytest.raises(ValueError):
        mfilth([])


def test_the_matched_filter_peaks_where_the_template_sits():
    g = [1.0, 2.0, 3.0, 2.0, 1.0]
    h = mfilth(g)["h"]
    x = [0.0] * 8 + g + [0.0] * 8
    y = [sum(h[k] * x[i - k] for k in range(len(h)) if i - k >= 0)
         for i in range(len(x))]
    peak = max(range(len(y)), key=lambda i: y[i])
    assert peak == 8 + len(g) - 1


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsafilt import (rangayyan_butterworth_lp,
                                  rangayyan_ch3_butterworth_pole_positions,
                                  rangayyan_hann_window)
    assert rangayyan_ch3_butterworth_pole_positions(
        BOOK_WC, 4)["n_left_half_plane"] == 4
    assert rangayyan_butterworth_lp(100.0, order=4, fs=1000.0)["kind"] == \
        "lowpass"
    assert rangayyan_hann_window(9)["endpoints"] == pytest.approx([0.0, 0.0])
