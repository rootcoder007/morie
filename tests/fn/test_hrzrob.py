"""Tests for hrzrob.horowitz_rate_beta_estimation (Horowitz eq. 2.26)."""

import math
import statistics

import pytest

from morie.fn.hrzrob import horowitz_rate_beta_estimation


SIZES = [100, 200, 400, 800, 1600]


def test_hrzrob_basic():
    """An exact n^{-1/2} law gives exponent -0.5, zero gap, zero SE and
    R^2 = 1; with perturbed errors the exponent is the least-squares
    slope of log error on log n."""
    r = horowitz_rate_beta_estimation([3.0 / math.sqrt(n) for n in SIZES], SIZES)
    assert r["exponent"] == pytest.approx(-0.5, abs=1e-12)
    assert r["gap"] == pytest.approx(0.0, abs=1e-12)
    assert r["se"] == pytest.approx(0.0, abs=1e-9)
    assert r["rsq"] == pytest.approx(1.0, abs=1e-12)
    assert r["intercept"] == pytest.approx(math.log(3.0), abs=1e-12)
    err = [3.0 / math.sqrt(n) * (1 + 0.1 * math.sin(n)) for n in SIZES]
    fit = statistics.linear_regression([math.log(n) for n in SIZES], [math.log(e) for e in err])
    s = horowitz_rate_beta_estimation(err, SIZES)
    assert s["exponent"] == pytest.approx(fit.slope, rel=1e-12)
    assert (s["k"], s["n"]) == (5, 1600)


def test_hrzrob_edge():
    """Non-positive errors or fewer than three points raise."""
    with pytest.raises(ValueError):
        horowitz_rate_beta_estimation([0.1, 0.0, 0.05], [10, 20, 40])
    with pytest.raises(ValueError):
        horowitz_rate_beta_estimation([0.1, 0.05], [10, 20])
