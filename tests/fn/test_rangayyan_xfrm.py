"""Rangayyan transforms (bsaxfrm): Chapter 3.4 and the homomorphic
equations of Chapter 4.7.

Expected values are hand-computed from the printed equations or are
closed forms of the test signal, never read back from the implementation.
"""

import cmath
import math

import pytest

from morie.fn.bsaxfrm import (circconv, clogsum, ctft, ctftf, dft, dftconv,
                              dftk, dftri, dfttw, dftx, dtft, dtftz, euler,
                              evenodd, evenpart, fourier, ftconv, ictft,
                              idftri,
                              logft, logmaxph, logminph, logseries, oddpart,
                              twidconj, twidcs, twiddle, twidper, ztconv,
                              ztrans)


# --------------------------------------------------------- z-transform

def test_ztrans_eq354_is_the_polynomial_in_z_inverse():
    # X(z) = 1 + 2 z^-1 + 3 z^-2 at z = 2  ->  1 + 1 + 0.75
    r = ztrans([1.0, 2.0, 3.0], z=2.0)
    assert r["X"] == pytest.approx(1.0 + 1.0 + 0.75)
    assert r["degree"] == 2 and r["causal"] is True


def test_ztrans_at_z_one_is_the_plain_sum():
    assert ztrans([1.0, -2.0, 4.0], z=1.0)["X"] == pytest.approx(3.0)


def test_ztrans_two_sided_uses_the_index_grid():
    # n0 = -1: X(z) = x(-1) z^1 + x(0) + x(1) z^-1
    r = ztrans([5.0, 1.0, 2.0], z=2.0, n0=-1)
    assert r["X"] == pytest.approx(5.0 * 2.0 + 1.0 + 2.0 / 2.0)
    assert r["causal"] is False


def test_ztrans_rejects_the_pole_at_zero():
    with pytest.raises(ValueError):
        ztrans([1.0, 2.0], z=0.0)


def test_ztconv_eq356_holds():
    r = ztconv([1.0, 2.0], [3.0, 4.0], z=1.7)
    assert r["y"] == pytest.approx([3.0, 10.0, 8.0])
    assert r["holds"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-12)


def test_dtftz_eq366_lands_on_the_unit_circle():
    r = dtftz([1.0, 2.0, 3.0], omega=0.7)
    assert r["on_unit_circle"] is True
    want = sum(v * cmath.exp(-1j * 0.7 * n)
               for n, v in enumerate([1.0, 2.0, 3.0]))
    assert r["X"] == pytest.approx(want)


def test_dtftz_uses_T_when_fs_is_given():
    x = [1.0, 2.0]
    r = dtftz(x, omega=2.0, fs=4.0)          # T = 0.25
    want = 1.0 + 2.0 * cmath.exp(-1j * 2.0 * 0.25)
    assert r["X"] == pytest.approx(want)
    assert r["T"] == pytest.approx(0.25)


# ------------------------------------------------- Fourier, eqs 3.74-3.78

def test_euler_eq374():
    r = euler(math.pi, 1.0)
    assert r["value"] == pytest.approx(-1.0 + 0j, abs=1e-15)
    assert r["real"] == pytest.approx(-1.0)
    assert r["imag"] == pytest.approx(0.0, abs=1e-15)
    assert r["unit_modulus"] is True


def test_ctft_eq375_at_dc_is_the_signal_integral():
    # rectangular pulse of height 1 on [0, 2]: X(0) = 2
    x = [1.0] * 201
    t = [i / 100.0 for i in range(201)]
    assert ctft(x, t=t, omega=0.0)["X"].real == pytest.approx(2.0)


def test_ctft_omega_and_f_forms_agree_at_omega_equals_two_pi_f():
    x = [1.0, 0.5, -0.25, 0.75, 0.0]
    t = [0.0, 0.25, 0.5, 0.75, 1.0]
    f0 = 0.7
    a = ctft(x, t=t, omega=2 * math.pi * f0)["X"]
    b = ctft(x, t=t, f=f0)["X"]
    assert a == pytest.approx(b)
    assert ctftf(x, f0, t=t)["X"] == pytest.approx(b)
    assert fourier(x, t=t, f=f0)["X"] == pytest.approx(b)


def test_ctft_reports_f_in_hz_for_the_omega_form():
    r = ctft([1.0, 1.0, 1.0], omega=2 * math.pi, dt=0.5)
    assert r["f"] == pytest.approx(1.0)
    assert r["variable"] == "omega"


def test_ctft_needs_exactly_one_frequency_variable():
    with pytest.raises(ValueError):
        ctft([1.0, 2.0])
    with pytest.raises(ValueError):
        ctft([1.0, 2.0], omega=1.0, f=1.0)


def test_ictft_eq377_carries_the_two_pi_only_in_the_omega_form():
    # a constant spectrum X = 1 on a symmetric band
    grid = [i / 200.0 - 1.0 for i in range(401)]
    X = [1.0 + 0j] * 401
    om = ictft(X, t=0.0, omega=grid)
    hz = ictft(X, t=0.0, f=grid)
    assert om["scale"] == pytest.approx(1.0 / (2 * math.pi))
    assert hz["scale"] == pytest.approx(1.0)
    assert om["x"].real == pytest.approx(2.0 / (2 * math.pi))
    assert hz["x"].real == pytest.approx(2.0)


def test_ictft_rejects_a_mismatched_grid():
    with pytest.raises(ValueError):
        ictft([1.0, 2.0], t=0.0, omega=[0.0, 1.0, 2.0])


def test_dtft_eq378_at_zero_is_the_sum():
    x = [1.0, -2.0, 3.0]
    assert dtft(x, 0.0)["X"].real == pytest.approx(2.0)


def test_dtft_is_two_pi_periodic():
    x = [1.0, -2.0, 3.0, 0.5]
    a = dtft(x, 0.9)["X"]
    b = dtft(x, 0.9 + 2 * math.pi)["X"]
    assert a == pytest.approx(b, abs=1e-12)


# ------------------------------------------------------ DFT, eqs 3.79-3.90

FOUR = [1.0, 2.0, 3.0, 4.0]
FOUR_DFT = [10 + 0j, -2 + 2j, -2 + 0j, -2 - 2j]


def test_dft_eq380_matches_the_hand_computed_transform():
    r = dft(FOUR)
    for got, want in zip(r["X"], FOUR_DFT):
        assert got == pytest.approx(want, abs=1e-12)
    assert r["conjugate_symmetric"] is True


def test_dft_of_an_impulse_is_flat():
    assert dft([1.0, 0.0, 0.0, 0.0])["X"] == pytest.approx([1 + 0j] * 4)


def test_dftk_eq379_reduces_to_the_dft_when_K_equals_N():
    got = dftk(FOUR, 4)
    assert got["X"] == pytest.approx(FOUR_DFT, abs=1e-12)
    assert got["aliased"] is False


def test_dftk_with_K_greater_than_N_interpolates_the_dtft():
    r = dftk(FOUR, 8)
    assert r["K"] == 8 and len(r["X"]) == 8
    # every K-point sample must equal the DTFT at the same normalized freq
    for k in range(8):
        want = dtft(FOUR, 2 * math.pi * k / 8)["X"]
        assert r["X"][k] == pytest.approx(want, abs=1e-10)


def test_dftk_flags_aliasing_when_K_is_too_small():
    assert dftk(FOUR, 2)["aliased"] is True


def test_dftx_puts_the_frequency_axis_in_hz():
    r = dftx(FOUR, fs=8.0)
    assert r["freqs"] == pytest.approx([0.0, 2.0, 4.0, 6.0])
    assert r["folding_frequency"] == pytest.approx(4.0)
    assert r["unique_bins"] == 3


def test_twiddle_eq382():
    w = twiddle(4)["W"]
    assert w == pytest.approx(-1j, abs=1e-15)
    assert twiddle(4, 2)["W"] == pytest.approx(-1.0 + 0j, abs=1e-15)
    assert twiddle(4)["root_of_unity"] is True


def test_dfttw_eq383_agrees_with_the_definition():
    r = dfttw(FOUR)
    assert r["X"] == pytest.approx(FOUR_DFT, abs=1e-10)
    assert r["agrees_with_definition"] is True


def test_twidcs_eq384_has_a_minus_sign_on_the_sine():
    r = twidcs(8, 1, 1)
    ang = 2 * math.pi / 8
    assert r["cos"] == pytest.approx(math.cos(ang))
    assert r["sin"] == pytest.approx(math.sin(ang))
    assert r["W"].imag == pytest.approx(-math.sin(ang))


def test_dftri_eq385_imag_is_minus_the_sine_projection():
    r = dftri(FOUR)
    assert r["X"] == pytest.approx(FOUR_DFT, abs=1e-12)
    for i, s in enumerate(r["sin_projection"]):
        assert r["imag"][i] == pytest.approx(-s)


def test_idftri_eq386_inverts_the_dft():
    r = idftri(FOUR_DFT)
    assert r["x"] == pytest.approx(FOUR)
    assert r["max_imaginary"] == pytest.approx(0.0, abs=1e-12)


def test_idftri_reports_a_residue_for_a_non_symmetric_spectrum():
    # a spectrum that no real signal could have
    r = idftri([1 + 0j, 1 + 1j, 0j, 0j])
    assert r["max_imaginary"] > 1e-6


def test_dftconv_eq387_needs_padding_for_linear_convolution():
    r = dftconv([1.0, 2.0], [3.0, 4.0])
    assert r["linear"] == pytest.approx([3.0, 10.0, 8.0])
    assert r["padded_length"] == 3
    assert r["from_dft"] == pytest.approx([3.0, 10.0, 8.0])
    assert r["holds"] is True
    # unpadded, the same product gives the circular result, which wraps
    assert r["circular"] == pytest.approx([11.0, 10.0])
    assert r["wraps_if_unpadded"] is True


def test_circconv_eq390_direct_and_via_dft_agree():
    r = circconv([1.0, 2.0], [3.0, 4.0])
    # y(0) = x0 h0 + x1 h1 = 3 + 8; y(1) = x0 h1 + x1 h0 = 4 + 6
    assert r["y"] == pytest.approx([11.0, 10.0])
    assert r["via_dft"] == pytest.approx([11.0, 10.0])
    assert r["agrees"] is True
    assert r["equals_linear"] is False


def test_circconv_equals_linear_once_N_is_large_enough():
    r = circconv([1.0, 2.0], [3.0, 4.0], npoints=3)
    assert r["y"] == pytest.approx([3.0, 10.0, 8.0])
    assert r["equals_linear"] is True


def test_circconv_rejects_a_period_shorter_than_the_signals():
    with pytest.raises(ValueError):
        circconv([1.0, 2.0, 3.0], [1.0], npoints=2)


def test_twidconj_eq388():
    r = twidconj(8, 3, 2)
    assert r["holds"] is True
    assert r["difference"] == pytest.approx(0.0, abs=1e-12)


def test_twidper_eq389():
    r = twidper(8, 3, 2)
    assert r["holds"] is True
    assert r["base"] == pytest.approx(r["shift_k"], abs=1e-9)
    assert r["base"] == pytest.approx(r["shift_n"], abs=1e-9)


# ------------------------------------------------- even/odd, eqs 3.92-3.94

def test_evenodd_eqs392_to_394():
    x = [1.0, 2.0, 3.0]
    n = [-1, 0, 1]
    r = evenodd(x, n)
    assert r["even"] == pytest.approx([2.0, 2.0, 2.0])
    assert r["odd"] == pytest.approx([-1.0, 0.0, 1.0])
    assert r["reconstruction_error"] == pytest.approx(0.0, abs=1e-15)


def test_odd_part_vanishes_at_the_origin():
    r = oddpart([4.0, 7.0, -1.0, 0.0, 2.0], n=[-2, -1, 0, 1, 2])
    assert r["odd"][2] == pytest.approx(0.0)


def test_evenpart_of_an_even_signal_is_itself():
    x = [3.0, 5.0, 3.0]
    r = evenpart(x, n=[-1, 0, 1])
    assert r["even"] == pytest.approx(x)
    assert r["odd"] == pytest.approx([0.0, 0.0, 0.0])


def test_evenodd_centres_an_odd_length_sequence_when_no_grid_is_given():
    r = evenodd([1.0, 2.0, 3.0])
    assert r["n"] == [-1, 0, 1]


def test_evenodd_refuses_an_even_length_sequence_without_a_grid():
    with pytest.raises(ValueError):
        evenodd([1.0, 2.0])


def test_evenodd_refuses_an_asymmetric_grid():
    with pytest.raises(ValueError):
        evenodd([1.0, 2.0, 3.0], n=[0, 1, 2])


# ------------------------------------------ homomorphic, eqs 4.58-4.71

def test_logft_eqs458_460_log_spectra_add():
    x = [1.0, 2.0, 3.0, 2.0, 1.0]
    p = [2.0, 2.0, 4.0, 4.0, 8.0]
    r = logft(x, p, omega=1.3, dt=0.5)
    assert r["y"] == pytest.approx([2.0, 4.0, 12.0, 8.0, 8.0])
    assert r["additive"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-12)


def test_logft_rejects_a_zero_factor():
    with pytest.raises(ValueError):
        logft([1.0, 0.0], [1.0, 1.0], omega=0.0)


def test_ftconv_eqs461_462():
    r = ftconv([1.0, 2.0, 1.0], [1.0, -1.0], omega=0.9, dt=0.25)
    assert r["y"] == pytest.approx([0.25, 0.25, -0.25, -0.25])
    assert r["holds"] is True


def test_clogsum_eq465_magnitudes_add_exactly():
    r = clogsum([1.0, 0.5], [1.0, -0.3], z=1.4)
    assert r["magnitude_difference"] == pytest.approx(0.0, abs=1e-12)
    assert r["holds_up_to_branch"] is True


def test_clogsum_reports_an_integer_branch_offset():
    # the principal phase can wrap; the offset must stay an exact integer
    r = clogsum([1.0, -0.9, 0.8], [1.0, -0.95, 0.9], z=complex(0.2, 0.98))
    off = r["branch_offset"]
    assert abs(off - round(off)) == pytest.approx(0.0, abs=1e-9)


def test_clogsum_rejects_a_zero_transform():
    with pytest.raises(ValueError):
        clogsum([1.0, -1.0], [1.0], z=1.0)      # X(1) = 0


def test_logseries_eq469_converges_to_the_logarithm():
    r = logseries(0.5, terms=60)
    assert r["value"].real == pytest.approx(math.log(1.5), abs=1e-12)
    assert r["error"] < 1e-12


def test_logseries_truncation_stays_inside_its_bound():
    r = logseries(0.5, terms=5)
    assert r["error"] <= r["error_bound"]


def test_logseries_refuses_outside_the_radius_of_convergence():
    with pytest.raises(ValueError):
        logseries(1.5)
    with pytest.raises(ValueError):
        logseries(-1.0)


def test_logminph_eq470_is_causal_and_sums_to_the_closed_form():
    r = logminph(0.5, terms=80, z=2.0)
    assert r["causal"] is True
    assert r["quefrency"][0] == 1
    assert r["coefficients"][0] == pytest.approx(-0.5)
    assert r["coefficients"][1] == pytest.approx(-0.125)   # -alpha^2/2
    assert r["error"] < 1e-12


def test_logminph_needs_z_outside_alpha():
    with pytest.raises(ValueError):
        logminph(0.5, z=0.25)


def test_logmaxph_eq471_is_anticausal():
    r = logmaxph(0.5, terms=80, z=0.5)
    assert r["causal"] is False
    assert r["quefrency"][0] == -1
    assert r["coefficients"][0] == pytest.approx(-0.5)
    assert r["error"] < 1e-12


def test_logmaxph_needs_z_inside_one_over_beta():
    with pytest.raises(ValueError):
        logmaxph(0.5, z=2.5)


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsaxfrm import (rangayyan_ch3_dft_definition,
                                  rangayyan_ch3_even_part,
                                  rangayyan_circular_conv_dft)
    assert rangayyan_ch3_dft_definition(FOUR)["X"] == pytest.approx(
        FOUR_DFT, abs=1e-12)
    assert rangayyan_circular_conv_dft([1.0, 2.0], [3.0, 4.0])["y"] == \
        pytest.approx([11.0, 10.0])
    assert rangayyan_ch3_even_part([3.0, 5.0, 3.0], n=[-1, 0, 1])["odd"] == \
        pytest.approx([0.0, 0.0, 0.0])
