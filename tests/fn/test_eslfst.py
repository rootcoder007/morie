"""Tests for eslfst.esl_f_test (ESL eq. 3.13 nested-model F).

Anchored on ESL Exercise 3.1: the F statistic for dropping a single
coefficient equals the square of that coefficient's t statistic.  Both
sides are computed here without going through esl_f_test twice.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.eslfst import esl_f_test

N = 40
_I = list(range(N))
X0 = [0.5 + 0.11 * i + 0.3 * ((i * i) % 7) for i in _I]
X1 = [-1.0 + 0.07 * i - 0.2 * ((i * i) % 7) for i in _I]
X2 = [2.0 - 0.03 * i + 0.15 * ((i * i * i) % 11) for i in _I]
Y = [1.0 + 2.0 * X0[i] - 0.5 * X1[i] + 0.35 * X2[i] + 0.2 * ((i * 3) % 5) for i in _I]
X = np.column_stack([np.asarray(X0), np.asarray(X1), np.asarray(X2)])


def _t_stat_of_last_column():
    D = np.column_stack([np.ones(N), X])
    y = np.asarray(Y, dtype=float)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    r = y - D @ beta
    s2 = float(r @ r) / (N - D.shape[1])
    v = s2 * float(np.linalg.inv(D.T @ D)[3, 3])
    return float(beta[3]) / v**0.5


def test_f_for_one_dropped_term_equals_t_squared():
    """ESL Exercise 3.1."""
    f = esl_f_test([0, 1], [0, 1, 2], X, Y)
    t = _t_stat_of_last_column()
    assert f["df1"] == 1
    assert f["df2"] == N - 4
    assert f["statistic"] == pytest.approx(t * t, rel=1e-10)


def test_bigger_model_never_has_larger_rss():
    f = esl_f_test([0], [0, 1, 2], X, Y)
    assert f["rss1"] <= f["rss0"]
    assert f["statistic"] > 0.0
    assert 0.0 <= f["p_value"] <= 1.0


def test_identical_models_are_rejected_as_not_nested_strictly():
    with pytest.raises(ValueError):
        esl_f_test([0, 1, 2], [0, 1, 2], X, Y)
    with pytest.raises(ValueError):
        esl_f_test([0, 1, 2], [0], X, Y)
