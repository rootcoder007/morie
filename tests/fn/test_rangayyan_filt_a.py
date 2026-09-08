"""Rangayyan filter design (bsafilt A1): LSI combination, the Laplace
route, the generic IIR and pole-zero responses, the moving average, the
Hann filter, and the order-statistic filters of Section 3.8.

Expected values are hand-computed from the printed equations.
"""

import math

import pytest

from morie.fn.bsafilt import (hannfilt, hannfr, hannfrs, hannimp, hannmag,
                              hannph, hanntf, hannz, iirdiff, iirtf,
                              laplace, laplacefr, lsiparh, lsiserh, mafir,
                              matf, osfilt, pzmag, pzphase)


# ------------------------------------------------------ LSI combination

def test_lsiserh_eq345_convolves_and_lengthens():
    r = lsiserh([1, 1], [1, 1])
    assert r["h"] == pytest.approx([1.0, 2.0, 1.0])
    assert r["n_taps"] == 3                       # 2 + 2 - 1
    assert lsiserh([1, 2, 3], [1])["h"] == pytest.approx([1.0, 2.0, 3.0])


def test_lsiserh_commutes():
    a, b = [1, 2, 3], [0.5, -1.0]
    assert lsiserh(a, b)["h"] == pytest.approx(lsiserh(b, a)["h"])
    assert lsiserh(a, b)["commutes"] is True


def test_lsiparh_eq349_adds_and_keeps_the_longer_length():
    r = lsiparh([1, 2], [10, 20, 30])
    assert r["h"] == pytest.approx([11.0, 22.0, 30.0])
    assert r["n_taps"] == 3                       # max, not the sum
    assert r["length_is_the_longer_branch"] is True


def test_lsi_combination_indexes_and_refuses_a_negative_index():
    assert lsiserh([1, 1], [1, 1], n=1)["value"] == pytest.approx(2.0)
    assert lsiparh([1, 2], [3], n=5)["value"] == 0.0
    with pytest.raises(ValueError):
        lsiserh([1, 1], [1, 1], n=-1)
    with pytest.raises(ValueError):
        lsiparh([], [1])


# ------------------------------------------------------- Laplace route

def test_laplace_eq350_of_a_unit_pulse():
    # h(t) = 1 on [0, 1]; H(0) = integral = 1
    t = [i / 200.0 for i in range(201)]
    h = [1.0] * 201
    assert laplace(h, t, 0.0)["H"].real == pytest.approx(1.0, abs=1e-12)
    # H(s) = (1 - e^-s)/s; at s = 1 that is 0.6321205588
    got = laplace(h, t, 1.0)["H"].real
    assert got == pytest.approx((1.0 - math.exp(-1.0)) / 1.0, abs=1e-5)


def test_laplace_reports_the_interval_it_integrated_over():
    t = [0.0, 0.5, 1.0]
    r = laplace([1.0, 1.0, 1.0], t, 0.0)
    assert (r["t_min"], r["t_max"]) == (0.0, 1.0)
    assert r["over_the_sampled_interval_only"] is True
    with pytest.raises(ValueError):
        laplace([1.0, 1.0], [1.0, 0.0], 0.0)       # t not increasing
    with pytest.raises(ValueError):
        laplace([1.0], [0.0], 0.0)                 # one sample


def test_laplacefr_eq352_matches_laplace_on_the_imaginary_axis():
    t = [i / 100.0 for i in range(101)]
    h = [math.exp(-2.0 * v) for v in t]
    w = 3.0
    a = laplacefr(h, w, t=t)["H"]
    b = laplace(h, t, complex(0.0, w))["H"]
    assert a.real == pytest.approx(b.real, abs=1e-12)
    assert a.imag == pytest.approx(b.imag, abs=1e-12)


def test_laplacefr_builds_the_grid_from_a_duration():
    h = [1.0] * 101
    r = laplacefr(h, 0.0, T=1.0)
    assert r["H"].real == pytest.approx(1.0, abs=1e-12)
    assert r["valid_only_inside_the_roc"] is True
    with pytest.raises(ValueError):
        laplacefr(h, 0.0)                          # neither t nor T
    with pytest.raises(ValueError):
        laplacefr(h, 0.0, T=-1.0)


# ------------------------------------------------- generic IIR responses

def test_iirtf_eq367_leading_one_is_implicit():
    # H(z) = 1 / (1 - 0.5 z^-1); at z = 1 that is 2
    r = iirtf([1.0], [-0.5], 1.0)
    assert r["H"].real == pytest.approx(2.0)
    assert r["denominator"] == pytest.approx([1.0, -0.5])
    assert r["leading_one_is_implicit"] is True


def test_iirtf_refuses_a_pole_and_a_wrong_order():
    with pytest.raises(ValueError):
        iirtf([1.0], [-1.0], 1.0)                  # z = 1 is the pole
    with pytest.raises(ValueError):
        iirtf([1.0, 2.0], [], 1.0, N=5)
    with pytest.raises(ValueError):
        iirtf([1.0], [0.5], 1.0, M=3)


def test_iirdiff_eq368_subtracts_the_feedback():
    # y(n) = x(n) + 0.5 y(n-1) is a_1 = -0.5
    r = iirdiff([1.0, 0.0, 0.0, 0.0], [1.0], [-0.5])
    assert r["y"] == pytest.approx([1.0, 0.5, 0.25, 0.125])
    assert r["feedback_is_subtracted"] is True
    assert r["recursive"] is True


def test_iirdiff_with_no_feedback_is_an_fir_filter():
    r = iirdiff([1.0, 2.0, 3.0], [0.5, 0.5])
    assert r["y"] == pytest.approx([0.5, 1.5, 2.5])
    assert r["recursive"] is False
    assert r["M"] == 0


def test_iirdiff_matches_the_transfer_function_at_dc():
    b, a = [1.0, 0.5], [-0.25]
    steps = [1.0] * 60
    y = iirdiff(steps, b, a)["y"]
    dc = iirtf(b, a, 1.0)["H"].real
    assert y[-1] == pytest.approx(dc, abs=1e-9)


# ------------------------------------------------------ pole-zero reading

def test_pzmag_eq372_is_the_ratio_of_distance_products():
    r = pzmag([2.0, 3.0], [1.0, 2.0])
    assert r["magnitude"] == pytest.approx(3.0)
    assert r["zero_product"] == pytest.approx(6.0)
    assert r["pole_product"] == pytest.approx(2.0)


def test_pzmag_flags_a_zero_and_refuses_a_pole_underfoot():
    assert pzmag([0.0], [1.0])["magnitude"] == 0.0
    assert pzmag([0.0], [1.0])["on_a_zero"] is True
    with pytest.raises(ValueError):
        pzmag([1.0], [0.0])
    with pytest.raises(ValueError):
        pzmag([-1.0], [1.0])


def test_pzphase_eq373_keeps_the_origin_term():
    # N = 1 zero, M = 2 poles, so (M - N) angle(z_0) does not vanish
    r = pzphase(complex(0.0, 1.0), [0.2], [0.1, 0.3])
    assert r["origin_term"] == pytest.approx(math.pi / 2.0)
    assert r["phase"] == pytest.approx(math.pi / 2.0 + 0.2 - 0.4)
    assert r["origin_term_vanishes_when_orders_match"] is False


def test_pzphase_origin_term_vanishes_when_the_orders_match():
    r = pzphase(complex(0.0, 1.0), [0.2, 0.1], [0.1, 0.3])
    assert r["origin_term"] == 0.0
    assert r["origin_term_vanishes_when_orders_match"] is True
    with pytest.raises(ValueError):
        pzphase(0.0, [0.1], [0.2])


# ------------------------------------------------------- moving average

def test_mafir_defaults_to_the_equal_weight_boxcar():
    r = mafir([1.0, 2.0, 3.0, 4.0], N=1)
    assert r["b"] == pytest.approx([0.5, 0.5])
    assert r["y"] == pytest.approx([0.5, 1.5, 2.5, 3.5])
    assert r["equal_weights"] is True
    assert r["delay_samples"] == pytest.approx(0.5)


def test_mafir_passes_a_constant_unchanged():
    r = mafir([5.0] * 20, N=4)
    assert r["y"][4:] == pytest.approx([5.0] * 16)
    assert r["dc_gain"] == pytest.approx(1.0)


def test_mafir_needs_coefficients_or_an_order():
    with pytest.raises(ValueError):
        mafir([1.0, 2.0])
    with pytest.raises(ValueError):
        mafir([1.0, 2.0], b_k=[1.0], N=7)


def test_matf_eq399_is_a_polynomial_with_no_poles():
    r = matf([0.25, 0.5, 0.25], 1.0)
    assert r["H"].real == pytest.approx(1.0)
    assert r["dc_gain"] == pytest.approx(1.0)
    assert r["always_stable"] is True
    with pytest.raises(ValueError):
        matf([1.0], 0.0)


# ------------------------------------------------------- the Hann filter

def test_hannfilt_eq3100_is_the_one_two_one_smoother():
    r = hannfilt([0.0, 0.0, 4.0, 0.0, 0.0])
    assert r["taps"] == pytest.approx([0.25, 0.5, 0.25])
    assert r["y"] == pytest.approx([0.0, 0.0, 1.0, 2.0, 1.0])
    assert r["delay_samples"] == pytest.approx(1.0)


def test_hannfilt_passes_a_constant_and_indexes():
    r = hannfilt([3.0] * 8)
    assert r["y"][2:] == pytest.approx([3.0] * 6)
    assert hannfilt([1.0, 2.0, 3.0], n=2)["value"] == pytest.approx(
        0.25 * (3.0 + 4.0 + 1.0))
    with pytest.raises(ValueError):
        hannfilt([1.0], n=9)


def test_hannimp_eq3101_is_finite_and_sums_to_one():
    r = hannimp()
    assert r["h"] == pytest.approx([0.25, 0.5, 0.25])
    assert r["sum"] == pytest.approx(1.0)
    assert r["finite"] is True
    assert hannimp(n=5)["value"] == 0.0


def test_hannimp_is_the_response_to_a_unit_impulse():
    y = hannfilt([1.0, 0.0, 0.0, 0.0])["y"]
    assert y[:3] == pytest.approx(hannimp()["h"])


def test_hannz_eq3102_factors_out_the_input():
    a = hannz(2.0, 1.0)
    b = hannz(5.0, 1.0)
    assert a["H"] == pytest.approx(b["H"])
    assert a["Y"].real == pytest.approx(2.0 * a["H"].real)
    assert a["transfer_function_is_input_independent"] is True


def test_hanntf_eq3103_has_a_double_zero_at_nyquist():
    r = hanntf(-1.0)
    assert abs(r["H"]) == pytest.approx(0.0, abs=1e-15)
    assert r["zeros"] == [-1.0, -1.0]
    assert r["zero_multiplicity"] == 2
    assert hanntf(1.0)["H"].real == pytest.approx(1.0)
    with pytest.raises(ValueError):
        hanntf(0.0)


def test_hannfr_eq3104_agrees_with_the_transfer_function():
    for w in (0.0, 0.3, 1.0, math.pi):
        z = complex(math.cos(w), math.sin(w))
        assert hannfr(w)["H"] == pytest.approx(hanntf(z)["H"], abs=1e-12)


def test_hannfrs_eq3105_agrees_with_the_raw_form():
    r = hannfrs([0.0, 0.4, 1.2, math.pi])
    assert r["agrees_with_raw_form"] is True
    assert r["max_difference_from_eq_3_104"] < 1e-12
    assert r["linear_phase"] is True


def test_hannmag_eq3106_is_lowpass():
    assert hannmag(0.0)["magnitude"] == pytest.approx(1.0)
    assert hannmag(math.pi)["magnitude"] == pytest.approx(0.0, abs=1e-15)
    vals = hannmag([0.0, 0.5, 1.0, 2.0, math.pi])["magnitude"]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_hannmag_matches_the_modulus_of_the_frequency_response():
    for w in (0.0, 0.7, 2.0, math.pi):
        assert hannmag(w)["magnitude"] == pytest.approx(abs(hannfr(w)["H"]),
                                                        abs=1e-12)


def test_hannph_eq3107_is_exactly_linear():
    r = hannph([0.0, 0.5, 1.0])
    assert r["phase"] == pytest.approx([0.0, -0.5, -1.0])
    assert r["group_delay"] == pytest.approx(1.0)
    assert r["constant_group_delay"] is True


def test_hannph_matches_the_argument_of_the_frequency_response():
    for w in (0.2, 1.0, 2.5):
        got = hannfr(w)["H"]
        assert math.atan2(got.imag, got.real) == pytest.approx(
            hannph(w)["phase"], abs=1e-12)


# --------------------------------------- order-statistic filters, Sec 3.8

SPIKY = [1, 1, 9, 1, 1, 1, -9, 1, 1]


def test_the_median_filter_removes_impulses_of_both_signs():
    r = osfilt(SPIKY, 3)
    assert r["y"] == pytest.approx([1.0] * 9)
    assert r["kind"] == "median"
    assert r["nonlinear"] is True
    assert r["no_frequency_response"] is True


def test_the_min_filter_removes_the_high_valued_impulse():
    y = osfilt(SPIKY, 3, kind="min")["y"]
    assert 9.0 not in y                    # the book's stated use
    assert -9.0 in y                       # and it cannot touch the low one


def test_the_max_filter_removes_the_low_valued_impulse():
    y = osfilt(SPIKY, 3, kind="max")["y"]
    assert -9.0 not in y
    assert 9.0 in y


def test_the_minmax_filter_applies_them_in_sequence():
    a = osfilt(SPIKY, 3, kind="min")["y"]
    b = osfilt(a, 3, kind="max")["y"]
    assert osfilt(SPIKY, 3, kind="minmax")["y"] == pytest.approx(b)


def test_order_one_is_the_min_and_order_w_is_the_max():
    assert osfilt(SPIKY, 3, kind="order", order=1)["y"] == pytest.approx(
        osfilt(SPIKY, 3, kind="min")["y"])
    assert osfilt(SPIKY, 3, kind="order", order=3)["y"] == pytest.approx(
        osfilt(SPIKY, 3, kind="max")["y"])
    mid = osfilt(SPIKY, 5, kind="order", order=3)["y"]
    assert mid == pytest.approx(osfilt(SPIKY, 5)["y"])


def test_alpha_near_one_half_approaches_the_median():
    heavy = osfilt(SPIKY, 5, kind="trimmed", alpha=0.4)["y"]
    assert heavy == pytest.approx(osfilt(SPIKY, 5)["y"])
    assert osfilt(SPIKY, 5, kind="trimmed", alpha=0.4)[
        "trimmed_each_end"] == 2


def test_alpha_zero_is_the_plain_moving_mean():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    flat = osfilt(x, 3, kind="trimmed", alpha=0.0)["y"]
    equal = osfilt(x, 3, kind="l", weights=[1, 1, 1])["y"]
    assert flat == pytest.approx(equal)


def test_the_l_filter_can_reproduce_the_median():
    y = osfilt(SPIKY, 3, kind="l", weights=[0, 1, 0])["y"]
    assert y == pytest.approx(osfilt(SPIKY, 3)["y"])


def test_the_output_is_the_same_length_and_the_edges_are_reflected():
    r = osfilt([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert r["n"] == 5
    assert r["edges"] == "symmetric reflection"
    # a monotone ramp survives a median filter untouched
    assert r["y"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_bad_order_statistic_arguments_are_refused():
    with pytest.raises(ValueError):
        osfilt(SPIKY, 4)                            # even window
    with pytest.raises(ValueError):
        osfilt(SPIKY, 99)                           # longer than the record
    with pytest.raises(ValueError):
        osfilt(SPIKY, 3, kind="bogus")
    with pytest.raises(ValueError):
        osfilt(SPIKY, 3, kind="trimmed", alpha=0.5)
    with pytest.raises(ValueError):
        osfilt(SPIKY, 3, kind="l")                  # no weights
    with pytest.raises(ValueError):
        osfilt(SPIKY, 3, kind="l", weights=[1, 1])  # wrong count
    with pytest.raises(ValueError):
        osfilt(SPIKY, 3, kind="order")              # no rank
    with pytest.raises(ValueError):
        osfilt(SPIKY, 3, kind="order", order=4)


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsafilt import (rangayyan_ch3_hann_filter,
                                  rangayyan_ch3_lsi_series_combined_h,
                                  rangayyan_order_stat_flt)
    assert rangayyan_ch3_lsi_series_combined_h([1, 1], [1, 1])["h"] == \
        pytest.approx([1.0, 2.0, 1.0])
    assert rangayyan_ch3_hann_filter([1.0, 2.0, 3.0])["taps"] == \
        pytest.approx([0.25, 0.5, 0.25])
    assert rangayyan_order_stat_flt(SPIKY, 3)["y"] == pytest.approx(
        [1.0] * 9)
