"""Rangayyan signals and systems (bsasig): linear convolution, LSI
interconnections, modulation.  Expected values hand-computed from the
printed equations.
"""

import math

import pytest

from morie.fn.bsasig import (amsig, fmsig, linconv, lsipar, lsipar2, lsipary,
                             lsiser, lsisery, ltiprod, perconv, tvlsi)


def test_linconv_eqs336_339():
    r = linconv([1.0, 2.0], [3.0, 4.0])
    assert r["y"] == pytest.approx([3.0, 10.0, 8.0])
    assert r["n"] == 3
    assert r["commutes"] is True


def test_linconv_contributions_sum_to_the_output():
    # eq (3.39): y is the sum of delayed, weighted copies of h
    r = linconv([1.0, -2.0, 0.5], [2.0, 1.0])
    for k in range(r["n"]):
        assert sum(row[k] for row in r["contributions"]) == pytest.approx(
            r["y"][k])


def test_linconv_with_an_impulse_returns_h():
    assert linconv([1.0], [5.0, -1.0, 2.0])["y"] == pytest.approx(
        [5.0, -1.0, 2.0])


def test_linconv_rejects_empty():
    with pytest.raises(ValueError):
        linconv([], [1.0])


def test_lsiser_eqs343_345():
    x = [1.0, 2.0, 3.0]
    h1 = [1.0, 1.0]
    h2 = [1.0, -1.0]
    r = lsiser(x, h1, h2)
    assert r["s"] == pytest.approx([1.0, 3.0, 5.0, 3.0])     # x * h1
    assert r["h"] == pytest.approx([1.0, 0.0, -1.0])         # h1 * h2
    assert r["equivalent"] is True
    # y = x * h computed independently here
    want = [1.0, 2.0, 2.0, -2.0, -3.0]
    assert r["y"] == pytest.approx(want)


def test_lsisery_is_the_same_cascade_output():
    x = [1.0, 2.0, 3.0]
    a = lsiser(x, [1.0, 1.0], [1.0, -1.0])
    b = lsisery(x, [1.0, 1.0], [1.0, -1.0])
    assert b["y"] == pytest.approx(a["y"])
    assert b["h"] == pytest.approx(a["h"])


def test_lsipar_eqs346_349():
    x = [1.0, 2.0]
    h1 = [1.0, 1.0]
    h2 = [2.0]
    r = lsipar(x, h1, h2)
    assert r["s1"] == pytest.approx([1.0, 3.0, 2.0])
    assert r["s2"] == pytest.approx([2.0, 4.0])
    assert r["h"] == pytest.approx([3.0, 1.0])     # h1 + h2, zero-extended
    assert r["y"] == pytest.approx([3.0, 7.0, 2.0])
    assert r["equivalent"] is True


def test_lsipar_zero_extends_the_shorter_branch():
    # the tail of the longer filter must survive the addition
    r = lsipar([1.0], [1.0, 2.0, 3.0], [1.0])
    assert r["h"] == pytest.approx([2.0, 2.0, 3.0])


def test_lsipar2_is_the_second_branch():
    r = lsipar2([1.0, 2.0], [2.0])
    assert r["s2"] == pytest.approx([2.0, 4.0])


def test_lsipary_matches_lsipar():
    a = lsipar([1.0, 2.0], [1.0, 1.0], [2.0])
    b = lsipary([1.0, 2.0], [1.0, 1.0], [2.0])
    assert b["y"] == pytest.approx(a["y"])
    assert b["h"] == pytest.approx(a["h"])


def test_series_convolves_where_parallel_adds():
    h1, h2 = [1.0, 1.0], [1.0, -1.0]
    ser = lsiser([1.0], h1, h2)["h"]
    par = lsipar([1.0], h1, h2)["h"]
    assert ser == pytest.approx([1.0, 0.0, -1.0])   # convolution, length 3
    assert par == pytest.approx([2.0, 0.0])         # addition, length 2


def test_ltiprod_eq353_in_the_s_domain():
    r = ltiprod([1.0, 2.0, 1.0], [1.0, -1.0], s=complex(0.3, 1.1), dt=0.5)
    assert r["holds"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-9)


def test_ltiprod_omega_form_is_s_on_the_imaginary_axis():
    x, h = [1.0, 0.5, -0.25], [1.0, 1.0]
    a = ltiprod(x, h, omega=1.7, dt=0.25)["Y"]
    b = ltiprod(x, h, s=complex(0.0, 1.7), dt=0.25)["Y"]
    assert a == pytest.approx(b)


def test_ltiprod_needs_exactly_one_variable():
    with pytest.raises(ValueError):
        ltiprod([1.0], [1.0])
    with pytest.raises(ValueError):
        ltiprod([1.0], [1.0], s=1.0, omega=1.0)


def test_perconv_eq390_is_the_circular_convolution():
    assert perconv([1.0, 2.0], [3.0, 4.0])["y"] == pytest.approx(
        [11.0, 10.0])
    assert perconv([1.0, 2.0], [3.0, 4.0], npoints=3)["y"] == pytest.approx(
        [3.0, 10.0, 8.0])


def test_amsig_uses_the_books_suppressed_carrier_model():
    # y = x cos(wc t): a constant modulator gives the bare carrier
    x = [1.0] * 8
    r = amsig(x, fc=1.0, fs=8.0)
    assert r["suppressed_carrier"] is True
    assert r["y"] == pytest.approx(r["carrier"])
    # x = 0 must give y = 0 -- the conventional model would not
    z = amsig([0.0] * 8, fc=1.0, fs=8.0)
    assert z["y"] == pytest.approx([0.0] * 8)


def test_amsig_conventional_model_keeps_the_carrier_through_a_zero_signal():
    z = amsig([0.0] * 8, fc=1.0, fs=8.0, conventional=True)
    assert z["y"] == pytest.approx(z["carrier"])


def test_amsig_synchronous_demodulation_recovers_half_the_signal():
    # x_d = x cos^2 = x/2 + (x/2) cos(2 wc t); the mean of cos^2 is 1/2
    n = 4096
    fs, fc = 1000.0, 100.0
    x = [1.0] * n
    r = amsig(x, fc=fc, fs=fs)
    assert sum(r["demodulated"]) / n == pytest.approx(0.5, abs=1e-3)
    assert r["image_frequency"] == pytest.approx(200.0)


def test_amsig_rejects_a_carrier_above_nyquist():
    with pytest.raises(ValueError):
        amsig([1.0, 2.0], fc=600.0, fs=1000.0)


def test_fmsig_instantaneous_frequency_follows_the_modulator():
    m = [0.0] * 10 + [50.0] * 10
    r = fmsig(m, fc=100.0, fs=1000.0, kf=1.0)
    assert r["instantaneous_frequency"][0] == pytest.approx(100.0)
    assert r["instantaneous_frequency"][-1] == pytest.approx(150.0)
    assert r["max_instantaneous_frequency"] == pytest.approx(150.0)
    assert r["aliases"] is False


def test_fmsig_with_a_zero_modulator_is_a_plain_cosine():
    n = 64
    r = fmsig([0.0] * n, fc=100.0, fs=1000.0)
    want = [math.cos(2 * math.pi * 100.0 * i / 1000.0) for i in range(n)]
    assert r["y"] == pytest.approx(want, abs=1e-12)


def test_fmsig_phase_uses_the_trapezoidal_integral():
    # constant modulator m: phase increment per sample is
    # 2 pi (fc + kf m) / fs exactly, which a plain running sum would miss
    # by half a step at the first sample
    m = [2.0] * 5
    r = fmsig(m, fc=10.0, fs=100.0, kf=1.0)
    step = 2 * math.pi * 12.0 / 100.0
    assert r["phase"][2] - r["phase"][1] == pytest.approx(step)


def test_fmsig_flags_aliasing():
    r = fmsig([400.0] * 4, fc=100.0, fs=1000.0)
    assert r["aliases"] is True


def test_tvlsi_reduces_to_lsi_when_the_kernel_is_constant():
    x = [1.0, 2.0, 3.0, 4.0]
    h = [1.0, 0.5]
    r = tvlsi(x, h)
    assert r["shift_invariant"] is True
    # y(n) = x(n) + 0.5 x(n-1), truncated at the record end
    assert r["y"] == pytest.approx([1.0, 2.5, 4.0, 5.5])


def test_tvlsi_uses_a_different_response_at_each_instant():
    x = [1.0, 1.0, 1.0]
    h = [[1.0], [2.0], [3.0]]
    r = tvlsi(x, h)
    assert r["y"] == pytest.approx([1.0, 2.0, 3.0])
    assert r["shift_invariant"] is False


def test_tvlsi_rejects_a_kernel_of_the_wrong_length():
    with pytest.raises(ValueError):
        tvlsi([1.0, 2.0, 3.0], [[1.0], [1.0]])


def test_pre_policy_spellings_still_resolve():
    from morie.fn.bsasig import (rangayyan_am_signal,
                                 rangayyan_ch3_lsi_parallel_total,
                                 rangayyan_linear_convolution)
    assert rangayyan_linear_convolution([1.0, 2.0], [3.0, 4.0])["y"] == \
        pytest.approx([3.0, 10.0, 8.0])
    assert rangayyan_ch3_lsi_parallel_total([1.0], [1.0], [2.0])["h"] == \
        pytest.approx([3.0])
    assert rangayyan_am_signal([0.0] * 4, fc=1.0, fs=8.0)["y"] == \
        pytest.approx([0.0] * 4)
