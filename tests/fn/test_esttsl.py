"""Tests for esttsl.theta_method (Hyndman-Billah form of the theta method).

Reference numbers are R forecast 8.x output, reproduced by
``thetaf(ts(y), h)`` and ``ses(ts(y), h, alpha = 0.3, initial = "simple")``
on the series below.
"""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.esttsl import _ses_given_alpha, theta_method

Y = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170]
THETAF_Y = [170.755338497375, 171.51060165527, 172.265864813165, 173.021127971059]
SES_SIMPLE_03 = 152.600028370006
B0 = 1.51052631578948


def test_esttsl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = theta_method(y)
    assert isinstance(result, dict)
    assert "forecast" in result


def test_esttsl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = theta_method(y)
    assert isinstance(result, dict)


def test_fixed_alpha_matches_r_ses_and_the_drift_formula():
    r = theta_method(Y, horizon=3, alpha=0.3, level0=112.0)
    assert abs(r["level"] - SES_SIMPLE_03) <= 1e-12 * SES_SIMPLE_03
    assert abs(r["linear_slope"] - B0) <= 1e-12
    damp = (1 - 0.7**20) / 0.3
    for h, f in enumerate(r["forecast"], 1):
        assert abs(f - (SES_SIMPLE_03 + B0 / 2 * (h - 1 + damp))) <= 1e-12 * f


def test_estimated_forecast_matches_forecast_thetaf():
    r = theta_method(Y, horizon=4)
    for ours, ref in zip(r["forecast"], THETAF_Y):
        # alpha sits on ets's upper bound 0.9999 here, so both fits agree;
        # 1e-9 covers the reference's printed 15 significant digits
        assert abs(ours - ref) <= 1e-9 * ref


def test_alpha_minimises_the_profiled_sse():
    y = [50 + 0.4 * t + 6 * math.sin(1.7 * t) + 3 * math.cos(0.9 * t**1.3) for t in range(1, 41)]
    r = theta_method(y, horizon=1)
    a = r["alpha"]
    assert r["sse"] <= _ses_given_alpha(y, a - 1e-6)[0]
    assert r["sse"] <= _ses_given_alpha(y, a + 1e-6)[0]
    # and it is below the point R's ets(A,N,N) Nelder-Mead stops at
    assert r["sse"] < _ses_given_alpha(y, 0.17724249875283404, 52.788849559525772)[0]


def test_theta_one_is_ses_and_theta_below_one_rejected():
    r = theta_method(Y, horizon=2, theta=1.0, alpha=0.3, level0=112.0)
    assert r["forecast"] == [r["level"], r["level"]]
    with pytest.raises(ValueError, match="theta must be at least 1"):
        theta_method(Y, theta=0.5)


def test_docstring_example():
    r = theta_method([1.0, 2.0, 4.0, 5.0], horizon=2, alpha=1.0)
    assert [round(v, 10) for v in r["forecast"]] == [5.7, 6.4]
