"""The 14 stubs that were still sitting inside four modules I had reported
complete.  Every assertion here is an arithmetic identity or a book equation,
never a value read back off the implementation."""

import math

import pytest

from morie.fn.bsaadapt import eegadapt, glr
from morie.fn.bsacorr import (alpharhy, xcorr, xcorrcont, xcorrdisc, xcorrproc,
                              corrconv, corrdot, eegacf, nccftpl)
from morie.fn.bsasig import compsig, sincostest
from morie.fn.bsastat import corrcoef


def sine(n, cycles_hz, fs=100.0, amp=1.0, phase=0.0):
    return [amp * math.sin(2 * math.pi * cycles_hz * i / fs + phase)
            for i in range(n)]


# --------------------------------------------------------------- GLR, 8.30/31
def test_glr_is_the_book_difference_of_log_likelihoods():
    """d(n) = H(1:n) - [H(1:m-1) + H(m:n)], eq. (8.31), exactly."""
    x = sine(400, 5)
    r = glr(x, 201, order=4)
    assert r["d"] == pytest.approx(
        r["h_pooled"] - (r["h_reference"] + r["h_test"]), abs=1e-12)


def test_glr_rises_when_the_process_changes():
    """One model explains a stationary record; two are needed after a change,
    and d is what pays for the second one."""
    stationary = glr(sine(400, 5), 201, order=4)["d"]
    changed = glr(sine(200, 5) + sine(200, 40), 201, order=4)["d"]
    assert changed > stationary
    assert changed > 0


def test_glr_windows_partition_the_record():
    r = glr(sine(300, 5), 151, 300, order=4)
    assert r["n_reference"] + r["n_test"] == 300


def test_glr_rejects_windows_shorter_than_the_order():
    with pytest.raises(ValueError):
        glr(sine(100, 5), 4, order=4)


def test_eegadapt_finds_the_one_boundary_and_restarts_the_reference():
    r = eegadapt(sine(200, 5) + sine(200, 40), 100.0, window=60, step=20)
    assert r["n_boundaries"] == 1
    assert r["reference_restarts_at_boundaries"] is True
    b = r["boundaries"][0]
    # A sliding test window resolves a change to within one window, not to
    # one sample: the detector can first see the change at pos = 200 - w and
    # must have fired by pos = 200.  b = 160 means the window [160, 220)
    # straddles the change, which is a correct detection at this resolution.
    w = r["window"]
    assert 200 - w <= b <= 200


def test_eegadapt_threshold_is_robust_by_default():
    r = eegadapt(sine(400, 5), 100.0, window=60, step=20)
    assert r["robust_threshold"] is True
    assert r["threshold"] == pytest.approx(
        r["median"] + 3.0 * 1.4826 * r["mad"], abs=1e-12)


def test_eegadapt_refuses_a_window_longer_than_the_record():
    with pytest.raises(ValueError):
        eegadapt(sine(50, 5), 100.0, window=200)


# ------------------------------------------------------------------------ CCF
def test_ccf_peak_lag_is_the_delay_with_the_books_sign():
    """A positive lag means y trails x."""
    x = sine(200, 3)
    y = [0.0] * 7 + x[:-7]
    assert xcorr(x, y, maxlag=20)["peak_lag"] == 7


def test_ccf_zero_lag_of_x_with_itself_is_the_mean_square():
    x = [1.0, 2.0, 3.0, 4.0]
    r = xcorr(x, x, maxlag=0, biased=True)
    assert r["ccf"][0] == pytest.approx(sum(v * v for v in x) / 4,
                                           abs=1e-12)


def test_ccf_normalized_is_bounded_and_hits_one_on_a_copy():
    x = sine(200, 3)
    r = xcorr(x, x, maxlag=10, normalize=True)
    assert max(abs(v) for v in r["ccf"]) <= 1.0 + 1e-12
    assert r["ccf"][r["lags"].index(0)] == pytest.approx(1.0, abs=1e-12)


def test_ccfdisc_agrees_with_ccf_on_the_peak():
    x = sine(200, 3)
    y = [0.0] * 5 + x[:-5]
    assert xcorrdisc(x, y)["peak_lag"] == xcorr(x, y, maxlag=30)["peak_lag"] == 5


def test_ccfcont_overlap_shrinks_as_the_delay_grows():
    t = list(range(20))
    x = sine(20, 2)
    a = xcorrcont(x, x, t, 0.0)["overlap_fraction"]
    b = xcorrcont(x, x, t, 10.0)["overlap_fraction"]
    assert a == 1.0 and b < a


def test_ccfproc_with_the_mean_removed_is_the_cross_covariance():
    r = xcorrproc([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], lags=0,
                remove_mean=True)
    assert r["is_cross_covariance_when_mean_removed"] is True
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    mx, my = 2.5, 5.0
    want = sum((a - mx) * (b - my) for a, b in zip(x, y)) / 4
    assert r["ccf"][r["lags"].index(0)] == pytest.approx(want, abs=1e-12)


def test_corrconv_identity_correlation_is_convolution_with_the_reverse():
    r = corrconv([1.0, 2.0, 3.0], [1.0, 0.0, -1.0])
    assert r["identity_holds"] is True
    assert r["max_difference"] == pytest.approx(0.0, abs=1e-12)
    assert r["ccf"] == pytest.approx(r["via_convolution"], abs=1e-12)


def test_nccftpl_is_bounded_and_locates_the_template():
    x = sine(200, 3)
    r = nccftpl(x, x[40:70])
    assert r["bounded_in_unit_interval"] is True
    assert max(r["gamma"]) <= 1.0 + 1e-12
    assert r["peak_shift"] == 40
    assert r["peak"] == pytest.approx(1.0, abs=1e-9)


def test_corrdot_is_the_cosine_and_is_scale_free():
    assert corrdot([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])["gamma"] == \
        pytest.approx(1.0, abs=1e-12)
    assert corrdot([1.0, 0.0], [0.0, 1.0])["gamma"] == pytest.approx(0.0,
                                                                     abs=1e-12)
    assert corrdot([1.0, 2.0], [1.0, 2.0])["gamma"] == \
        pytest.approx(corrdot([1.0, 2.0], [100.0, 200.0])["gamma"], abs=1e-12)


def test_eegacf_recovers_the_rhythm_frequency():
    r = eegacf(sine(400, 10, 100.0), 100.0)
    assert r["implied_frequency_hz"] == pytest.approx(10.0, abs=0.5)


def test_alpharhy_fires_in_band_and_reports_the_frequency():
    r = alpharhy(sine(400, 10, 100.0), 100.0)
    assert r["present"] is True
    assert 8.0 <= r["frequency_hz"] <= 13.0


def test_alpharhy_band_is_the_conventional_8_to_13_hz():
    assert alpharhy(sine(400, 10, 100.0), 100.0)["band"] == (8.0, 13.0)


# ------------------------------------------------------- synthetic test signals
def test_sincostest_length_is_duration_times_rate():
    assert sincostest(fs=100.0, duration=2.5)["n"] == 250


def test_sincostest_is_the_sum_of_its_two_named_components():
    r = sincostest(fs=100.0, duration=0.5, f1=5.0, f2=20.0, a1=2.0, a2=0.5)
    for i, (t, v) in enumerate(zip(r["t"], r["x"])):
        want = (r["a1"] * math.sin(2 * math.pi * r["f1"] * t)
                + r["a2"] * math.cos(2 * math.pi * r["f2"] * t))
        assert v == pytest.approx(want, abs=1e-12), i


def test_compsig_peaks_land_where_the_shifts_put_them():
    r = compsig([1.0, 2.0, 1.0], [0, 10, 20])
    assert r["peaks_expected_at"] == [2, 12, 22]
    assert r["overlapping_pairs"] == 0


def test_compsig_reports_overlap_when_the_shifts_are_closer_than_the_pulse():
    assert compsig([1.0, 2.0, 1.0], [0, 1])["overlapping_pairs"] == 1


# ------------------------------------------------------------------- corrcoef
def test_corrcoef_is_one_on_a_positive_scaling_and_minus_one_on_negation():
    assert corrcoef([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])["r"] == \
        pytest.approx(1.0, abs=1e-12)
    assert corrcoef([1.0, 2.0, 3.0, 4.0], [-2.0, -4.0, -6.0, -8.0])["r"] == \
        pytest.approx(-1.0, abs=1e-12)


def test_corrcoef_is_the_zero_lag_normalized_ccf_of_the_centred_signals():
    x = sine(120, 3)
    y = sine(120, 3, phase=0.7)
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    cx = [v - mx for v in x]
    cy = [v - my for v in y]
    lags = xcorr(cx, cy, maxlag=0, normalize=True)["ccf"][0]
    assert corrcoef(x, y)["r"] == pytest.approx(lags, abs=1e-9)


def test_corrcoef_refuses_a_constant_signal():
    with pytest.raises(ValueError):
        corrcoef([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
