"""Rangayyan optimal and adaptive filtering (bsaadapt): Wiener, ANC,
LMS, RLS, Kalman, adaptive segmentation.

Expected values are hand-computed from the printed equations.  The book
erratum at eq. (3.224) -- its first line has a minus where its own
second line and eq. (3.225) have a plus -- is pinned by a test, since
the minus form diverges.
"""

import math

import pytest

from morie.fn.bsaadapt import (abcdlemma, acfseg, anc, ancinput, ancout,
                               kalman, lmsdescent, lmsfilt, lmsout,
                               lmssqerr, lmsvarstep, lmszhang, msegrad,
                               pcgseg, psdacf, riccati, rlsapriori,
                               rlsfilt, rlslattice, rlsmonitor, rlsnormal,
                               rlsobj, rlsupdate, sem, whopf, widrowhoff,
                               wienerconv, wienerdot, wienerfilt,
                               wienerfreq, wienerfreqrel, wienerhopf,
                               wienermin, wieneropt, wienerout, wienersnr)


def sine(n, cycles, amp=1.0, phase=0.0):
    return [amp * math.sin(2 * math.pi * cycles * i / n + phase)
            for i in range(n)]


def lcg(n, seed=7, lo=-0.5, hi=0.5):
    out, s = [], seed
    for _ in range(n):
        s = (1103515245 * s + 12345) % 2147483648
        out.append(lo + (hi - lo) * (s / 2147483648.0))
    return out


# --------------------------------------------------- Wiener, eqs 3.154-3.176

def test_wienerout_eq3154_is_a_convolution():
    r = wienerout([1.0, 0.5], [1.0, 2.0, 3.0])
    assert r["d_hat"] == pytest.approx([1.0, 2.5, 4.0])
    assert r["settled_from"] == 1


def test_wienerdot_eq3155_matches_the_convolution_form():
    w = [1.0, 0.5]
    x = [1.0, 2.0, 3.0]
    conv = wienerout(w, x)["d_hat"]
    # x(n) runs backwards in time: [x(n), x(n-1)]
    assert wienerdot(w, [x[2], x[1]])["d_hat"] == pytest.approx(conv[2])


def test_wienerdot_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        wienerdot([1.0, 2.0], [1.0])


def test_msegrad_eq3167_vanishes_at_the_optimum():
    Phi = [[2.0, 1.0], [1.0, 2.0]]
    Theta = [3.0, 3.0]
    w = wieneropt(Phi, Theta)["w_o"]
    g = msegrad(Phi, Theta, w)
    assert g["gradient"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert g["at_optimum"] is True
    # away from it the gradient is -2 Theta + 2 Phi w
    g0 = msegrad(Phi, Theta, [0.0, 0.0])
    assert g0["gradient"] == pytest.approx([-6.0, -6.0])


def test_wienerhopf_eq3168_solves_the_normal_equation():
    r = wienerhopf([[2.0, 1.0], [1.0, 2.0]], [3.0, 3.0])
    assert r["w"] == pytest.approx([1.0, 1.0])
    assert r["max_residual"] == pytest.approx(0.0, abs=1e-12)


def test_wienerhopf_rejects_a_singular_matrix():
    with pytest.raises(ValueError):
        wienerhopf([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0])


def test_wieneropt_eq3169_solves_rather_than_inverting():
    r = wieneropt([[2.0, 1.0], [1.0, 2.0]], [3.0, 3.0])
    assert r["w_o"] == pytest.approx([1.0, 1.0])
    assert r["solved_not_inverted"] is True


def test_wienermin_eq3172_is_the_variance_less_the_explained_part():
    Phi = [[2.0, 1.0], [1.0, 2.0]]
    Theta = [3.0, 3.0]
    r = wienermin(Phi, Theta, var_d=10.0)
    assert r["explained"] == pytest.approx(6.0)     # Theta' w_o
    assert r["j_min"] == pytest.approx(4.0)
    assert r["consistent"] is True


def test_wienermin_flags_inconsistent_statistics():
    # a variance too small for the covariances cannot be right
    r = wienermin([[2.0, 1.0], [1.0, 2.0]], [3.0, 3.0], var_d=1.0)
    assert r["j_min"] < 0
    assert r["consistent"] is False


def test_wienerconv_eq3174_holds_at_the_solution():
    phi = [2.0, 1.0, 0.5]
    theta = [3.0, 3.0]
    Phi = [[phi[abs(i - j)] for j in range(2)] for i in range(2)]
    w = wieneropt(Phi, theta)["w_o"]
    r = wienerconv(w, phi, theta)
    assert r["holds"] is True
    assert r["requires_stationarity"] is True


def test_wienerfreqrel_eq3175_flags_undetermined_bins():
    W = [1.0, 0.5, 0.0]
    sxx = [2.0, 4.0, 0.0]
    sxd = [2.0, 2.0, 0.0]
    r = wienerfreqrel(W, sxx, sxd)
    assert r["holds"] is True
    assert r["undetermined_bins"] == [2]


def test_wienerfreq_eq3176_is_the_csd_over_the_psd():
    r = wienerfreq([2.0, 4.0], [1.0, 2.0])
    assert [v.real for v in r["W"]] == pytest.approx([0.5, 0.5])


def test_wienerfreq_zeroes_the_undetermined_bins():
    r = wienerfreq([1.0, 0.0], [1.0, 5.0])
    assert r["W"][1] == 0
    assert r["undetermined_bins"] == [1]
    assert r["zero_where_undetermined"] is True


def test_wienersnr_eq3186_has_the_three_stated_properties():
    sd = [0.0, 1.0, 4.0, 1.0]
    seta = [1.0, 0.0, 1.0, 9.0]
    r = wienersnr(sd, seta)
    assert r["W"][0] == 0.0                      # nothing to restore
    assert r["W"][1] == 1.0                      # noiseless
    assert r["W"][2] == pytest.approx(0.8)
    assert r["W"][3] == pytest.approx(0.1)
    assert r["W"][2] > r["W"][3]                 # falls with the SNR
    assert r["zero_where_signal_absent"] is True
    assert r["unity_where_noise_absent"] is True


def test_wienersnr_rejects_a_negative_psd():
    with pytest.raises(ValueError):
        wienersnr([1.0, -1.0], [1.0, 1.0])


def test_whopf_builds_a_toeplitz_system_from_data():
    n = 400
    x = sine(n, 7)
    d = [0.5 * v for v in x]
    r = whopf(x, d, order=3)
    assert r["toeplitz"] is True
    assert r["acf_biased"] is True
    assert len(r["w"]) == 3
    assert r["j_min"] < r["var_d"]


def test_whopf_needs_more_samples_than_taps():
    with pytest.raises(ValueError):
        whopf([1.0, 2.0], [1.0, 2.0], order=5)


def test_wienerfilt_needs_exactly_one_route():
    x = sine(64, 3)
    with pytest.raises(ValueError):
        wienerfilt(x)
    with pytest.raises(ValueError):
        wienerfilt(x, desired=x, sd=[1.0], seta=[1.0])


def test_wienerfilt_time_route_recovers_a_scaled_signal():
    n = 400
    x = sine(n, 7)
    d = [0.5 * v for v in x]
    r = wienerfilt(x, desired=d, order=3)
    assert r["route"] == "time"
    err = max(abs(a - b) for a, b in zip(r["y"][10:], d[10:]))
    assert err < 0.05


def test_wienerfilt_frequency_route_needs_both_psds():
    x = sine(32, 3)
    with pytest.raises(ValueError):
        wienerfilt(x, sd=[1.0] * 17)


def test_wienerfilt_frequency_route_suppresses_a_noisy_band():
    n = 64
    x = sine(n, 4)
    half = n // 2 + 1
    sd = [1.0 if k == 4 else 0.0 for k in range(half)]
    seta = [0.0 if k == 4 else 1.0 for k in range(half)]
    r = wienerfilt(x, sd=sd, seta=seta)
    assert r["route"] == "frequency"
    assert r["y"] == pytest.approx(x, abs=1e-9)


# ------------------------------------------------------- ANC, eqs 3.187-3.196

def test_ancinput_checks_the_independence_premise():
    v = sine(256, 3)
    m = sine(256, 41)
    r = ancinput(v, m)
    assert r["x"] == pytest.approx([a + b for a, b in zip(v, m)])
    assert r["independent"] is True
    bad = ancinput(v, [0.5 * u for u in v])
    assert bad["correlation"] == pytest.approx(1.0)
    assert bad["independent"] is False


def test_ancout_eq3196_makes_the_error_the_output():
    r = ancout([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    assert r["e"] == pytest.approx([0.5, 1.5, 2.5])
    assert r["v_hat"] == r["e"]
    assert r["error_is_the_output"] is True


def test_lmsout_eq3195_filters_the_reference():
    r = lmsout([1.0, 0.5], [2.0, 4.0])
    assert r["y"] == pytest.approx([2.0, 5.0])
    assert r["filters_the_reference"] is True


def test_lmssqerr_eq3200_expands_the_square():
    r = lmssqerr(3.0, [1.0, 2.0], [0.5, 0.25])
    assert r["e"] == pytest.approx(3.0 - 1.0)
    assert r["agrees"] is True
    assert r["nonnegative"] is True
    assert r["instantaneous_not_expected"] is True


def test_lmsdescent_eqs3201_3202_equal_widrow_hoff():
    w, e, r, mu = [0.1, -0.2], 0.7, [1.0, 2.0], 0.05
    a = lmsdescent(w, e, r, mu)
    b = widrowhoff(w, e, r, mu)
    assert a["w_next"] == pytest.approx(b["w_next"])
    assert a["gradient"] == pytest.approx([-2 * e * v for v in r])


def test_widrowhoff_eq3203_keeps_the_factor_of_two():
    r = widrowhoff([0.0, 0.0], 1.0, [1.0, 2.0], 0.1)
    assert r["w_next"] == pytest.approx([0.2, 0.4])
    assert r["factor_of_two_is_in_the_equation"] is True


def test_widrowhoff_reports_the_stability_bound():
    small = widrowhoff([0.0], 1.0, [1.0], 0.5)
    big = widrowhoff([0.0], 1.0, [1.0], 5.0)
    assert small["within_bound"] is True
    assert big["within_bound"] is False


def test_lmsvarstep_eq3204_is_eq3203_with_a_moving_mu():
    a = lmsvarstep([0.0, 0.0], 1.0, [1.0, 2.0], 0.1)
    b = widrowhoff([0.0, 0.0], 1.0, [1.0, 2.0], 0.1)
    assert a["w_next"] == pytest.approx(b["w_next"])
    assert a["time_varying"] is True


def test_lmszhang_eq3205_normalizes_by_the_running_power():
    r = lmszhang(0.5, 4, 2.0, alpha=0.02)
    assert r["power"] > 0
    assert r["mu"] == pytest.approx(0.5 / (5 * r["power"]))
    # a louder input gives a smaller step
    loud = lmszhang(0.5, 4, 20.0, alpha=0.02)
    assert loud["mu"] < r["mu"]


def test_lmszhang_enforces_the_books_ranges():
    with pytest.raises(ValueError):
        lmszhang(1.5, 4, 1.0)
    with pytest.raises(ValueError):
        lmszhang(0.5, 4, 1.0, alpha=0.9)


def test_lmsfilt_cancels_a_correlated_interference():
    n = 2000
    v = sine(n, 5, 1.0)
    ref = sine(n, 61, 1.0)
    m = [0.8 * u for u in ref]
    x = [a + b for a, b in zip(v, m)]
    r = lmsfilt(x, ref, order=4, mu=0.005)
    tail = range(n // 2, n)
    before = max(abs(x[i] - v[i]) for i in tail)
    after = max(abs(r["e"][i] - v[i]) for i in tail)
    assert after < before / 4
    assert r["within_bound"] is True
    assert r["converges_in_the_mean_only"] is True
    # the taps do not equal [0.8, 0, 0, 0]; what must equal 0.8 is the
    # filter's gain at the reference frequency
    w = math.radians(0.0) + 2 * math.pi * 61 / n
    re_ = sum(c * math.cos(k * w) for k, c in enumerate(r["final_weights"]))
    im = -sum(c * math.sin(k * w) for k, c in enumerate(r["final_weights"]))
    assert math.hypot(re_, im) == pytest.approx(0.8, abs=0.03)


def test_lms_misadjustment_grows_with_the_step_size():
    n = 2000
    v = sine(n, 5)
    ref = sine(n, 61)
    x = [a + 0.8 * b for a, b in zip(v, ref)]
    tail = range(n // 2, n)

    def resid(mu):
        r = lmsfilt(x, ref, order=4, mu=mu)
        return max(abs(r["e"][i] - v[i]) for i in tail)

    r = [resid(mu) for mu in (0.005, 0.02, 0.05)]
    assert r == sorted(r)


def test_lmsfilt_rejects_an_unstable_step_size():
    with pytest.raises(ValueError):
        lmsfilt(sine(64, 3), sine(64, 7), order=2, mu=-1.0)


def test_lmsfilt_variable_step_survives_a_reference_starting_at_zero():
    # sine(.., phase 0) has r(0) = 0; seeding the power recursion from
    # r(0)^2 would make mu(0) unbounded
    ref = sine(256, 41)
    assert ref[0] == 0.0
    r = lmsfilt(sine(256, 5), ref, order=4, mu=0.5, variable=True)
    assert r["variable_step"] is True
    assert len(r["step_history"]) == 256
    assert all(math.isfinite(v) and v > 0 for v in r["step_history"])


# ------------------------------------------------------ RLS, eqs 3.206-3.225

def test_rlsobj_eq3206_weights_recent_errors_more():
    e = [1.0, 1.0, 1.0]
    r = rlsobj(e, 0.5)
    assert r["weights"] == pytest.approx([0.25, 0.5, 1.0])
    assert r["xi"] == pytest.approx(1.75)
    assert r["memory"] == pytest.approx(2.0)


def test_rlsobj_lambda_one_is_a_growing_window():
    r = rlsobj([1.0, 1.0], 1.0)
    assert r["growing_window"] is True
    assert r["memory"] == math.inf


def test_rlsobj_enforces_the_lambda_range():
    with pytest.raises(ValueError):
        rlsobj([1.0], 1.5)


def test_rlsnormal_eq3207_has_the_wiener_hopf_form():
    r = rlsnormal([[2.0, 1.0], [1.0, 2.0]], [3.0, 3.0])
    assert r["w_tilde"] == pytest.approx([1.0, 1.0])
    assert r["same_form_as_wiener_hopf"] is True
    assert r["direct_inversion"] is True


def test_abcdlemma_eq3213_matches_the_direct_inverse():
    A = [[4.0, 1.0], [1.0, 3.0]]
    B = [[1.0], [2.0]]
    C = [[1.0]]
    D = [[1.0, 2.0]]
    r = abcdlemma(A, B, C, D)
    assert r["holds"] is True
    assert r["scalar_when_k_is_one"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-9)


def test_rlsupdate_eq3224_uses_the_plus_form():
    r = rlsupdate([1.0, 2.0], [0.5, 0.25], 2.0)
    assert r["w_next"] == pytest.approx([2.0, 2.5])
    assert r["sign"] == "+"
    assert "erratum" in r


def test_rlsapriori_eq3225_uses_the_previous_weights():
    r = rlsapriori(5.0, [1.0, 2.0], [0.5, 1.0])
    assert r["prediction"] == pytest.approx(2.5)
    assert r["alpha"] == pytest.approx(2.5)
    assert r["uses_previous_weights"] is True


def test_rlsfilt_converges_faster_than_lms():
    n = 600
    ref = sine(n, 31)
    x = [0.8 * v for v in ref]
    rls = rlsfilt(x, ref, order=3, lam=0.98)
    lms = lmsfilt(x, ref, order=3, mu=0.01)
    early = slice(20, 60)
    e_rls = max(abs(v) for v in rls["e"][early])
    e_lms = max(abs(v) for v in lms["e"][early])
    assert e_rls < e_lms


def test_rlsfilt_keeps_p_symmetric():
    r = rlsfilt(sine(400, 11), sine(400, 29), order=4, lam=0.99)
    assert r["p_symmetrized"] is True
    assert r["p_symmetry_error"] < 1e-6


def test_rlsfilt_enforces_the_lambda_range():
    with pytest.raises(ValueError):
        rlsfilt(sine(64, 3), sine(64, 7), order=2, lam=1.5)


def test_rlslattice_reports_every_order_and_its_stability():
    r = rlslattice(sine(500, 13), order=4)
    assert len(r["reflection"]) == 4
    assert r["stable"] is True
    assert r["every_stage_is_a_predictor"] is True
    assert all(abs(v) < 1.0 for v in r["reflection"])


def test_rlsmonitor_excludes_the_convergence_transient():
    n = 800
    x = sine(n, 7)[:n // 2] + sine(n, 61)[n // 2:]
    r = rlsmonitor(x, order=4, settle=100, window=40)
    assert r["transient_excluded"] is True
    assert r["settle"] == 100
    assert all(b >= 100 for b in r["boundaries"])


def test_rlsmonitor_finds_a_change_of_statistics():
    n = 800
    x = sine(n, 5)[:n // 2] + sine(n, 71)[n // 2:]
    r = rlsmonitor(x, order=4, settle=80, window=30, threshold=3.0)
    assert r["n_boundaries"] >= 1


# ------------------------------------------------------ Kalman and Riccati

def test_kalman_tracks_a_constant_state():
    F = [[1.0]]
    H = [[1.0]]
    Q = [[1e-6]]
    R = [[0.5]]
    truth = 3.0
    z = [[truth + v] for v in lcg(200, seed=11)]
    r = kalman(z, F, H, Q, R, x0=[0.0], P0=[[10.0]])
    assert r["states"][-1][0] == pytest.approx(truth, abs=0.15)
    assert r["covariances"][-1][0][0] < 10.0
    assert r["p_symmetrized"] is True
    assert r["joseph_form"] is False


def test_kalman_covariance_shrinks_monotonically_for_a_static_state():
    F, H, Q, R = [[1.0]], [[1.0]], [[0.0]], [[1.0]]
    z = [[1.0]] * 40
    r = kalman(z, F, H, Q, R, x0=[0.0], P0=[[5.0]])
    p = [c[0][0] for c in r["covariances"]]
    assert all(b <= a + 1e-12 for a, b in zip(p, p[1:]))


def test_kalman_rejects_a_wrong_shaped_measurement():
    with pytest.raises(ValueError):
        kalman([[1.0, 2.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]])


def test_riccati_is_the_fixed_point_of_the_kalman_recursion():
    F, H, Q, R = [[0.9]], [[1.0]], [[0.1]], [[1.0]]
    r = riccati(F, H, Q, R)
    assert r["converged"] is True
    # running the filter long enough must land on the same covariance
    z = [[0.0]] * 500
    k = kalman(z, F, H, Q, R, x0=[0.0], P0=[[1.0]])
    # the DARE solves for the PREDICTED covariance; the filter stores the
    # UPDATED one, P(1 - K H)
    post = r["P"][0][0] * (1.0 - r["K"][0][0])
    assert k["covariances"][-1][0][0] == pytest.approx(post, abs=1e-9)
    # and the scalar DARE has a closed form here: p^2 + 0.09 p - 0.1 = 0
    assert r["P"][0][0] == pytest.approx(
        (-0.09 + math.sqrt(0.09 ** 2 + 0.4)) / 2, abs=1e-9)
    assert r["steady_state_is_the_wiener_solution"] is True


# ---------------------------------------------------------- segmentation

def test_sem_is_scale_free():
    a = [1.0, 2.0, 4.0, 8.0]
    r_same = sem(a, a)
    r_gain = sem([3.0 * v for v in a], a)
    assert r_same["sem"] == pytest.approx(0.0)
    assert r_gain["shape_only"] == pytest.approx(0.0, abs=1e-12)
    assert r_gain["mean_offset"] == pytest.approx(math.log(3.0))
    assert r_gain["scale_free"] is True


def test_sem_grows_with_a_shape_change():
    a = [1.0, 2.0, 4.0, 8.0]
    b = [8.0, 4.0, 2.0, 1.0]
    assert sem(b, a)["shape_only"] > 0.5


def test_sem_rejects_a_negative_psd():
    with pytest.raises(ValueError):
        sem([1.0, -1.0], [1.0, 1.0])


def test_acfseg_eq827_power_distance_sees_a_gain_change():
    a = sine(200, 7)
    b = [5.0 * v for v in a]
    r = acfseg(b, a)
    # sqrt of the powers are in the ratio 5, so d_P = |5-1|/1 = 4
    assert r["power_distance"] == pytest.approx(4.0, abs=1e-9)
    assert r["amplitude_invariant"] is False
    assert r["boundary"] is True


def test_acfseg_eq828_spectral_distance_vanishes_on_an_identical_window():
    a = sine(200, 7)
    r = acfseg(a, a)
    assert r["power_distance"] == pytest.approx(0.0, abs=1e-12)
    assert r["spectral_distance"] == pytest.approx(0.0, abs=1e-12)
    assert r["distance"] == pytest.approx(0.0, abs=1e-12)
    assert r["boundary"] is False


def test_acfseg_spectral_distance_grows_when_the_shape_changes():
    a = sine(200, 5)
    b = sine(200, 47)
    assert acfseg(b, a)["spectral_distance"] > 0.1


def test_acfseg_eq829_weights_the_two_distances_by_their_thresholds():
    a = sine(200, 5)
    b = sine(200, 47)
    r = acfseg(b, a, thp=2.0, thf=4.0)
    assert r["distance"] == pytest.approx(
        r["power_distance"] / 2.0 + r["spectral_distance"] / 4.0)
    assert r["boundary"] is (r["distance"] > 1.0)


def test_acfseg_picks_q_where_the_acfs_first_turn_negative():
    a = sine(400, 8)
    r = acfseg(a, a)
    assert r["lags_auto"] is True
    assert r["lags"] >= 1
    # every ACF value inside the square roots of eq. (8.28) is nonnegative
    assert all(v >= 0 for v in r["acf_test"])
    assert all(v >= 0 for v in r["acf_reference"])


def test_acfseg_refuses_a_q_past_the_first_negative_lag():
    a = sine(400, 8)
    auto = acfseg(a, a)["lags"]
    assert acfseg(a, a, lags=auto)["lags"] == auto
    with pytest.raises(ValueError):
        acfseg(a, a, lags=auto + 1)


def test_acfseg_rejects_a_dead_window_and_a_bad_threshold():
    a = sine(200, 5)
    with pytest.raises(ValueError):
        acfseg([0.0] * 200, a)
    with pytest.raises(ValueError):
        acfseg(a, a, thp=0.0)


def test_pcgseg_restarts_the_reference_at_each_boundary():
    fs = 1000.0
    n = 1200
    x = sine(n, 20)[:n // 2] + sine(n, 120)[n // 2:]
    r = pcgseg(x, fs=fs, window=100, step=50, order=4)
    assert r["reference_restarted_at_boundaries"] is True
    assert r["robust_threshold"] is True
    assert len(r["sem"]) == len(r["times"])


def test_psdacf_eq430_agrees_with_the_circular_acf():
    r = psdacf(sine(64, 5))
    assert r["holds"] is True
    assert r["linear_acf_is_smoothed"] is True


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsaadapt import (rangayyan_ch3_widrow_hoff_lms,
                                   rangayyan_ch3_wiener_hopf_normal_equation,
                                   rangayyan_lms_filter)
    assert rangayyan_ch3_wiener_hopf_normal_equation(
        [[2.0, 1.0], [1.0, 2.0]], [3.0, 3.0])["w"] == pytest.approx(
            [1.0, 1.0])
    assert rangayyan_ch3_widrow_hoff_lms(
        [0.0], 1.0, [1.0], 0.1)["w_next"] == pytest.approx([0.2])
    assert rangayyan_lms_filter(sine(128, 3), sine(128, 29),
                                order=2)["order"] == 2
