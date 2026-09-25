"""Tests for bsaphys.rangayyan_coronary_ad (sec. 7.10, AR model)."""

import math

import pytest

from morie.fn.bsaphys import rangayyan_coronary_ad


FS = 2000.0
X = [math.sin(2 * math.pi * 100 * t / FS) + 0.2 * math.sin(2 * math.pi * 500 * t / FS)
     + 0.1 * math.sin(13.1 * t) for t in range(1024)]


def _acf(x, p):
    n = len(x)
    m = sum(x) / n
    xs = [v - m for v in x]
    return [sum(xs[i] * xs[i + k] for i in range(n - k)) / n for k in range(p + 1)]


def test_rgcorad_basic():
    """The AR coefficients solve the Yule-Walker equations of the biased
    autocorrelation, sum_k a_k r(|i-k|) = -r(i), i = 1..p, and the
    prediction error is r(0) + sum a_k r(k) (Levinson-Durbin)."""
    p = 8
    r = rangayyan_coronary_ad(X, FS, order=p)
    a = [float(v) for v in r["ar_coeffs"]]
    a = a[1:] if len(a) == p + 1 else a
    rr = _acf(X, p)
    for i in range(1, p + 1):
        lhs = sum(a[k - 1] * rr[abs(i - k)] for k in range(1, p + 1))
        assert lhs == pytest.approx(-rr[i], abs=1e-9)
    assert r["prediction_error"] == pytest.approx(rr[0] + sum(a[k - 1] * rr[k] for k in range(1, p + 1)), rel=1e-9)
    assert r["order"] == p


def test_rgcorad_edge():
    """A band above Nyquist raises."""
    with pytest.raises(ValueError):
        rangayyan_coronary_ad(X, 1000.0)
