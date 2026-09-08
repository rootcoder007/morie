"""Tests for bsacorr.rangayyan_acf_estimate, the unbiased ACF of Ch. 3.

The generated version of this file passed a 100-element random ARRAY as
`max_lag` -- an integer argument -- and then asserted only
`isinstance(result, dict)`.  It could not fail for any reason connected
to the autocorrelation.  These assert properties the estimator must have.
"""

import math

import pytest

from morie.fn.bsacorr import rangayyan_acf_estimate


def sine(n, hz, fs=100.0):
    return [math.sin(2 * math.pi * hz * i / fs) for i in range(n)]


def test_lag_zero_is_the_mean_square():
    """R_xx(0) = (1/N) sum x(n)^2 -- at lag 0 both the biased and the
    unbiased divisors are N, so the two estimators must agree there."""
    x = [1.0, 2.0, 3.0, 4.0]
    want = sum(v * v for v in x) / 4.0
    unb = rangayyan_acf_estimate(x, max_lag=0)
    bia = rangayyan_acf_estimate(x, max_lag=0, biased=True)
    # lags run 0..max_lag one-sided, so lag m sits at position m
    assert unb["acf"][0] == pytest.approx(want, abs=1e-12)
    assert bia["acf"][0] == pytest.approx(want, abs=1e-12)


def test_the_two_estimators_differ_by_exactly_the_divisor_ratio():
    """The only difference is N vs N-|m|, so biased(m)/unbiased(m) must
    be exactly (N-|m|)/N at every lag."""
    x = sine(64, 5)
    n = len(x)
    u = rangayyan_acf_estimate(x, max_lag=10)
    b = rangayyan_acf_estimate(x, max_lag=10, biased=True)
    for lag, uv, bv in zip(list(u["lags"]), list(u["acf"]),
                           list(b["acf"])):
        if abs(uv) < 1e-14:
            continue
        assert bv / uv == pytest.approx((n - abs(lag)) / n, abs=1e-12)


def test_both_estimator_variants_are_returned_and_acf_selects_one():
    """The payload carries acf_unbiased and acf_biased side by side, and
    `acf` is whichever the biased= flag selected -- so a caller can see
    the trade-off rather than having to recompute the other."""
    x = sine(64, 5)
    u = rangayyan_acf_estimate(x, max_lag=8)
    b = rangayyan_acf_estimate(x, max_lag=8, biased=True)
    assert list(u["acf"]) == list(u["acf_unbiased"])
    assert list(b["acf"]) == list(b["acf_biased"])
    # and the pair is the same in both calls -- only the selection moves
    assert list(u["acf_biased"]) == pytest.approx(list(b["acf_biased"]),
                                                  abs=1e-12)


def test_acf_recovers_a_known_period():
    """A 10 Hz tone at 100 Hz has a 10-sample period, so the ACF must
    peak again at lag 10."""
    r = rangayyan_acf_estimate(sine(400, 10, 100.0), max_lag=25)
    vals = list(r["acf"])
    positive = {m: vals[m] for m in range(1, 26)}
    assert max(positive, key=positive.get) == 10


def test_biased_estimate_is_bounded_by_lag_zero():
    """The biased form is non-negative definite, so |R(m)| <= R(0).
    (The unbiased form has no such guarantee -- that is its whole
    trade-off -- so this is asserted only of the biased one.)"""
    r = rangayyan_acf_estimate(sine(60, 3), max_lag=50, biased=True)
    vals = list(r["acf"])
    r0 = vals[0]
    assert all(abs(v) <= r0 + 1e-12 for v in vals)


def test_max_lag_must_be_an_integer_not_a_signal():
    """The generated test passed a 100-element array here; refusing it
    is correct behaviour and is pinned so it cannot silently regress."""
    x = sine(100, 5)
    with pytest.raises((ValueError, TypeError)):
        rangayyan_acf_estimate(x, max_lag=x)


def test_max_lag_out_of_range_is_refused():
    x = sine(20, 2)
    with pytest.raises(ValueError):
        rangayyan_acf_estimate(x, max_lag=20)
    with pytest.raises(ValueError):
        rangayyan_acf_estimate(x, max_lag=-1)
