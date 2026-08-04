"""Rangayyan parametric modelling (bsaar): Levinson-Durbin, LPC/AR,
all-pole PSD, order selection, pole-zero forms.

Expected values are hand-computed from the printed equations or are
exact properties of a synthetic AR process.
"""

import math

import pytest

from morie.fn.bsaar import (armafit, arfit, fpeorder, hrvar, hrvratio, levinson,
                            lpc, lpcsynth, mdlorder, pcgar, polezero, pzform,
                            pzformz, pzresp)


def ar1(n, a1=0.8, seed=7):
    """x(n) = a1 x(n-1) + e(n) with a deterministic pseudo-noise."""
    x, s = [0.0], seed
    for _ in range(n):
        s = (1103515245 * s + 12345) % 2147483648
        e = (s / 2147483648.0) - 0.5
        x.append(a1 * x[-1] + e)
    return x[1:]


# ------------------------------------------------- Levinson-Durbin 7.37-7.39

def test_levinson_first_order_matches_the_hand_recursion():
    # eq (7.37) at i = 1: gamma_1 = -phi(1)/phi(0); eq (7.39) follows
    r = levinson([1.0, 0.5])
    assert r["reflection"][0] == pytest.approx(-0.5)
    assert r["a"] == pytest.approx([-0.5])
    assert r["error"] == pytest.approx((1 - 0.25) * 1.0)


def test_levinson_second_order_by_hand():
    phi = [1.0, 0.5, 0.2]
    g1 = -phi[1] / phi[0]
    e1 = (1 - g1 * g1) * phi[0]
    g2 = -(phi[2] + g1 * phi[1]) / e1
    a2 = [g1 + g2 * g1, g2]
    r = levinson(phi)
    assert r["reflection"] == pytest.approx([g1, g2])
    assert r["a"] == pytest.approx(a2)
    assert r["error"] == pytest.approx((1 - g2 * g2) * e1)


def test_levinson_error_is_monotone_and_reports_stability():
    r = levinson([1.0, 0.6, 0.3, 0.1, 0.05])
    assert r["monotone"] is True
    assert r["stable"] is True
    assert all(a >= b for a, b in zip(r["errors"], r["errors"][1:]))


def test_levinson_flags_an_unstable_model():
    # a reflection coefficient of magnitude >= 1 is the book's stability test
    r = levinson([1.0, 2.0])
    assert abs(r["reflection"][0]) >= 1.0
    assert r["stable"] is False


def test_levinson_rejects_an_order_beyond_the_supplied_lags():
    with pytest.raises(ValueError):
        levinson([1.0, 0.5], order=3)


# ----------------------------------------------------------- LPC / AR fit

def test_lpc_recovers_a_known_ar1_with_the_books_sign():
    # x(n) = 0.8 x(n-1) + e means y~(n) = -a1 y(n-1), so a1 = -0.8
    r = lpc(ar1(4000, 0.8), 1)
    assert r["a"][0] == pytest.approx(-0.8, abs=0.05)
    assert r["sign_convention"].startswith("A(z) = 1 + sum a_k")
    assert r["stable"] is True


def test_lpc_gain_squared_is_the_prediction_error_eq735():
    x = ar1(2000, 0.6)
    r = lpc(x, 4)
    # eq (7.35): G^2 = eps_P = phi(0) + sum a_k phi(k)
    want = r["acf"][0] + sum(a * p for a, p in zip(r["a"], r["acf"][1:]))
    assert r["error"] == pytest.approx(want, rel=1e-9)
    assert r["gain"] ** 2 == pytest.approx(r["error"], rel=1e-9)


def test_lpc_residual_whitens_a_known_ar_process():
    x = ar1(4000, 0.85)
    r = lpc(x, 2)
    resid = r["residual"][2:]
    n = len(resid)
    mu = sum(resid) / n
    lag1 = sum((resid[i] - mu) * (resid[i + 1] - mu)
               for i in range(n - 1)) / n
    var = sum((v - mu) ** 2 for v in resid) / n
    assert abs(lag1 / var) < 0.1          # the input's lag-1 was 0.85


def test_lpc_refuses_the_covariance_method_rather_than_faking_it():
    with pytest.raises(ValueError):
        lpc(ar1(100), 2, method="covariance")


def test_lpc_needs_more_samples_than_the_order():
    with pytest.raises(ValueError):
        lpc([1.0, 2.0, 3.0], 5)


def test_lpcsynth_inverts_lpc_on_the_same_coefficients():
    x = ar1(500, 0.7)
    fit = lpc(x, 3)
    back = lpcsynth(fit["a"], fit["residual"])
    assert back["y"][:400] == pytest.approx(x[:400], abs=1e-9)
    assert back["diverged"] is False


def test_lpcsynth_flags_divergence_for_an_unstable_filter():
    # A(z) = 1 - 2.5 z^-1 has its pole at 2.5, outside the unit circle
    r = lpcsynth([-2.5], [1.0] + [0.0] * 2000)
    assert r["diverged"] is True


def test_flipping_the_sign_convention_moves_the_poles():
    # the observable consequence of using the other convention: A(z)
    # becomes 1 - sum a_k z^-k, a different polynomial with different
    # roots, so both the resonance frequencies and their bandwidths
    # change.  For A(z) = 1 + a1 z^-1 the pole moves from -a1 to +a1.
    a = lpc(ar1(4000, 0.8), 1)["a"]
    right = polezero([1.0], a)["poles"][0]
    flipped = polezero([1.0], [-v for v in a])["poles"][0]
    assert right == pytest.approx(-a[0] + 0j, abs=1e-9)
    assert flipped == pytest.approx(a[0] + 0j, abs=1e-9)
    assert abs(right - flipped) > 1.0


def test_lpcsynth_rejects_a_wrong_length_initial_state():
    with pytest.raises(ValueError):
        lpcsynth([0.5, 0.2], [1.0], initial=[0.0])


# ------------------------------------------------------------ AR spectrum

def test_arfit_psd_peaks_at_the_ar_resonance():
    fs = 1000.0
    # a conjugate pole pair at radius 0.95, angle 2 pi 100 / fs
    r0, w0 = 0.95, 2 * math.pi * 100.0 / fs
    a1 = -2 * r0 * math.cos(w0)
    a2 = r0 * r0
    x, hist = [], [0.0, 0.0]
    s = 3
    for _ in range(4000):
        s = (1103515245 * s + 12345) % 2147483648
        e = (s / 2147483648.0) - 0.5
        v = e - a1 * hist[0] - a2 * hist[1]
        hist = [v, hist[0]]
        x.append(v)
    r = arfit(x, 6, fs=fs, nfreq=512)
    peak = r["freqs"][max(range(len(r["psd"])),
                          key=lambda i: r["psd"][i])]
    assert peak == pytest.approx(100.0, abs=8.0)
    assert r["max_peaks"] == 3


def test_arfit_rejects_a_bad_sampling_rate():
    with pytest.raises(ValueError):
        arfit(ar1(200), 4, fs=0.0)


# --------------------------------------------------------- order criteria

def test_fpeorder_penalises_order():
    # errors barely improve past p = 2, so FPE must not choose the largest
    errs = [1.0, 0.5, 0.499, 0.4989, 0.49889]
    r = fpeorder(errs, n_samples=200)
    assert r["order"] == 2
    assert len(r["criterion"]) == 5


def test_fpeorder_formula_matches_akaike():
    errs = [1.0, 0.5]
    n = 100
    r = fpeorder(errs, n_samples=n)
    assert r["criterion"][0] == pytest.approx(1.0 * (n + 2) / (n - 2))
    assert r["criterion"][1] == pytest.approx(0.5 * (n + 3) / (n - 3))


def test_mdlorder_is_at_least_as_strict_as_aic():
    errs = [1.0, 0.5, 0.499, 0.4989, 0.49889]
    r = mdlorder(errs, n_samples=200)
    assert r["order"] <= r["aic_order"]
    assert r["stricter_than_aic"] is True
    assert r["penalty_per_parameter"] == pytest.approx(math.log(200))


def test_mdlorder_formula_matches_rissanen():
    r = mdlorder([1.0, 0.5], n_samples=64)
    assert r["criterion"][0] == pytest.approx(64 * math.log(1.0)
                                              + 1 * math.log(64))
    assert r["criterion"][1] == pytest.approx(64 * math.log(0.5)
                                              + 2 * math.log(64))


def test_order_criteria_reject_nonpositive_variances():
    with pytest.raises(ValueError):
        fpeorder([1.0, 0.0], 100)
    with pytest.raises(ValueError):
        mdlorder([1.0, -1.0], 100)


# --------------------------------------------------------- pole-zero forms

def test_pzform_eq369_evaluates_the_factored_form():
    r = pzform([0.5], [0.8], z=2.0)
    want = (1 - 0.5 / 2.0) / (1 - 0.8 / 2.0)
    assert r["H"] == pytest.approx(want)
    assert r["stable"] is True


def test_pzform_flags_an_unstable_pole():
    assert pzform([], [1.2])["stable"] is False


def test_pzform_rejects_z_zero():
    with pytest.raises(ValueError):
        pzform([0.5], [0.2], z=0.0)


def test_pzformz_eq370_agrees_with_eq369():
    zs, ps = [0.5, -0.3], [0.8]
    r = pzformz(zs, ps, z=complex(0.6, 0.7))
    assert r["agrees_with_eq369"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-12)
    assert r["exponent"] == len(ps) - len(zs)


def test_pzformz_exponent_is_M_minus_N():
    assert pzformz([0.1, 0.2, 0.3], [0.4])["exponent"] == -2


def test_pzresp_magnitude_is_the_ratio_of_distances_eq372():
    r = pzresp([0.5], [0.8], omega=0.9)
    z0 = complex(math.cos(0.9), math.sin(0.9))
    assert r["magnitude"] == pytest.approx(abs(z0 - 0.5) / abs(z0 - 0.8))
    assert r["magnitude_matches_product"] is True


def test_pzresp_zero_on_the_unit_circle_is_a_spectral_null():
    # a zero at exp(j w0) sends one distance to zero at w = w0
    w0 = 1.1
    z0 = complex(math.cos(w0), math.sin(w0))
    r = pzresp([z0], [0.0], omega=w0)
    assert r["magnitude"] == pytest.approx(0.0, abs=1e-12)
    assert r["zero_distances"][0] == pytest.approx(0.0, abs=1e-12)


def test_pzresp_pole_near_the_circle_is_a_resonance():
    w0 = 1.1
    near = 0.99 * complex(math.cos(w0), math.sin(w0))
    on = pzresp([], [near], omega=w0)["magnitude"]
    off = pzresp([], [near], omega=w0 + 0.5)["magnitude"]
    assert on > 5 * off


def test_polezero_finds_the_roots_of_H():
    r = polezero([1.0, -0.5], [-0.8])
    assert r["zeros"][0] == pytest.approx(0.5 + 0j, abs=1e-9)
    assert r["poles"][0] == pytest.approx(0.8 + 0j, abs=1e-9)
    assert r["stable"] is True
    assert r["minimum_phase"] is True


def test_polezero_conjugate_pair():
    # 1 - 1.2 z^-1 + 0.85 z^-2 has poles at 0.6 +/- j sqrt(0.85 - 0.36)
    r = polezero([1.0], [-1.2, 0.85])
    mags = sorted(abs(p) for p in r["poles"])
    assert mags[0] == pytest.approx(math.sqrt(0.85), abs=1e-6)
    assert mags[1] == pytest.approx(math.sqrt(0.85), abs=1e-6)


def test_polezero_flags_a_zero_on_the_unit_circle():
    r = polezero([1.0, -1.0], None)
    assert len(r["zeros_on_unit_circle"]) == 1
    assert r["minimum_phase"] is False


# --------------------------------------------------------------- ARMA, PCG

def test_armafit_returns_both_polynomials():
    r = armafit(ar1(1000, 0.7), p=2, q=1)
    assert len(r["a"]) == 2
    assert len(r["b"]) == 2
    assert r["two_stage"] is True


def test_armafit_rejects_a_negative_ma_order():
    with pytest.raises(ValueError):
        armafit(ar1(200), p=2, q=-1)


def test_pcgar_reports_resonances_once_per_conjugate_pair():
    fs = 1000.0
    n = 2000
    x = [math.sin(2 * math.pi * 60 * i / fs)
         + 0.4 * math.sin(2 * math.pi * 180 * i / fs) for i in range(n)]
    r = pcgar(x, fs=fs, order=8)
    freqs = [d["frequency"] for d in r["resonances"]]
    assert all(0.0 < f <= fs / 2 for f in freqs)
    assert len(r["resonances"]) <= 4
    assert any(abs(f - 60.0) < 15.0 for f in freqs)


def test_pcgar_default_order_scales_with_fs():
    x = ar1(600, 0.7)
    assert pcgar(x, fs=1000.0)["order"] < pcgar(x, fs=8000.0)["order"]


# ------------------------------------------------------------------- HRV

def rr_series(n=300, mean=0.8, lf=0.10, hf=0.25, amp_lf=0.02, amp_hf=0.02):
    rr, t = [], 0.0
    for _ in range(n):
        v = mean + amp_lf * math.sin(2 * math.pi * lf * t) \
            + amp_hf * math.sin(2 * math.pi * hf * t)
        rr.append(v)
        t += v
    return rr


def test_hrvar_bands_follow_the_task_force_edges():
    r = hrvar(rr_series(), order=12)
    assert r["bands"]["lf"] == (0.04, 0.15)
    assert r["bands"]["hf"] == (0.15, 0.40)
    assert r["mean_rr"] == pytest.approx(0.8, abs=0.02)


def test_hrvar_removes_the_mean_before_modelling():
    r = hrvar(rr_series(), order=12)
    assert abs(sum(r["resampled"]) / len(r["resampled"])) < 1e-9


def test_hrvratio_reports_components_not_only_the_ratio():
    r = hrvratio(rr_series(), order=12)
    assert r["lf"] > 0 and r["hf"] > 0
    assert r["lf_hf_ratio"] == pytest.approx(r["lf"] / r["hf"])
    assert r["lf_nu"] + r["hf_nu"] == pytest.approx(100.0)
    assert "sympathovagal" in r["interpretation_caveat"]


def test_hrvar_rejects_nonpositive_intervals():
    with pytest.raises(ValueError):
        hrvar([0.8, -0.1] * 8)


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsaar import (rangayyan_ar_order_mdl,
                                rangayyan_levinson_durbin,
                                rangayyan_pole_zero_plot)
    assert rangayyan_levinson_durbin([1.0, 0.5])["a"] == pytest.approx([-0.5])
    assert rangayyan_ar_order_mdl([1.0, 0.5], 64)["order"] in (1, 2)
    assert rangayyan_pole_zero_plot([1.0, -0.5])["zeros"][0] == \
        pytest.approx(0.5 + 0j, abs=1e-9)
