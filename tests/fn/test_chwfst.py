"""Tests for chwfst.chow_forecast_test (Chow 1960 forecast test).

Anchored on the Salkever equivalence: the Chow forecast F equals the F
for the joint significance of n2 observation dummies added to the
pooled regression.  That construction is built here from scratch and
does not call chow_forecast_test.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.chwfst import chow_forecast_test

N = 40
_I = list(range(N))
X0 = [0.5 + 0.11 * i + 0.3 * ((i * i) % 7) for i in _I]
X1 = [-1.0 + 0.07 * i - 0.2 * ((i * i) % 7) for i in _I]
X2 = [2.0 - 0.03 * i + 0.15 * ((i * i * i) % 11) for i in _I]
Y = [1.0 + 2.0 * X0[i] - 0.5 * X1[i] + 0.35 * X2[i] + 0.2 * ((i * 3) % 5) for i in _I]
X = np.column_stack([np.asarray(X0), np.asarray(X1), np.asarray(X2)])
SPLIT = 30


def _rss(D, y):
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ b
    return float(r @ r)


def test_chow_equals_observation_dummy_f():
    y = np.asarray(Y, dtype=float)
    D = np.column_stack([np.ones(N), X])
    n2 = N - SPLIT
    dummies = [np.array([1.0 if i == SPLIT + j else 0.0 for i in range(N)]) for j in range(n2)]
    Dd = np.column_stack([D] + dummies)
    f_dummy = ((_rss(D, y) - _rss(Dd, y)) / n2) / (_rss(Dd, y) / (N - D.shape[1] - n2))

    res = chow_forecast_test(Y, X, SPLIT)
    assert res["df1"] == n2
    assert res["df2"] == SPLIT - 4
    assert res["statistic"] == pytest.approx(f_dummy, rel=1e-9)


def test_chow_detects_a_planted_level_break():
    y = list(Y)
    broken = y[:SPLIT] + [v + 50.0 for v in y[SPLIT:]]
    quiet = chow_forecast_test(Y, X, SPLIT)
    loud = chow_forecast_test(broken, X, SPLIT)
    assert loud["statistic"] > quiet["statistic"]
    assert loud["p_value"] < 1e-6


def test_chow_needs_a_forecast_period():
    with pytest.raises(ValueError):
        chow_forecast_test(Y, X, N)
