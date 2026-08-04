"""Tests for glmsT.linear_trend (Kendall 1938 / Mann-Kendall).

Anchored on hand-computable limiting cases: a strictly increasing
series must give tau = 1 and S = n(n-1)/2; a strictly decreasing one
tau = -1; and tau-b must fall back to (P-Q)/(P+Q) when there are no
ties.
"""

import pytest

from morie.fn.glmsT import linear_trend

T = list(range(1, 21))


def test_strictly_increasing_series():
    res = linear_trend(T, [float(i) ** 3 for i in T])
    assert res["tau"] == pytest.approx(1.0)
    assert res["tau_a"] == pytest.approx(1.0)
    assert res["S"] == 190  # 20 * 19 / 2
    assert res["n_discordant"] == 0
    assert res["p_value"] < 1e-6


def test_strictly_decreasing_series():
    res = linear_trend(T, [-float(i) ** 3 for i in T])
    assert res["tau"] == pytest.approx(-1.0)
    assert res["S"] == -190
    assert res["n_concordant"] == 0


def test_tau_b_equals_tau_a_without_ties():
    res = linear_trend(T, [float((i * 7) % 20) for i in T])
    assert res["tau"] == pytest.approx(res["tau_a"])


def test_tau_b_differs_from_tau_a_with_ties():
    x = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]
    res = linear_trend(list(range(10)), x)
    assert res["tau"] != pytest.approx(res["tau_a"])
    assert abs(res["tau"]) <= 1.0


def test_flat_series_has_no_trend():
    res = linear_trend(T, [3.0] * 20)
    assert res["S"] == 0
    assert res["z"] == 0.0
    assert res["p_value"] == pytest.approx(1.0)
