"""Tests for dffits (scaled deletion influence).

Replaces the generated stub, which imported ``dffits`` from a module
whose function is called ``dffitsols``.
"""

import math

from morie.fn.dffits import dffitsols


def _data(n=20, noise=True):
    X = [[float(i)] for i in range(n)]
    y = [2.0 + 0.5 * i for i in range(n)]
    if noise:
        # an exact fit leaves zero residual variance, so the studentised
        # residual is 0/0; every real use has noise in it
        for i in range(n):
            y[i] += ((i * 7919) % 13 - 6) * 0.01
    return X, y


def test_leverages_sum_to_p():
    X, y = _data()
    res = dffitsols(X, y)
    assert abs(sum(res["leverage"]) - res["p"]) < 1e-9
    assert res["p"] == 2            # slope plus intercept
    assert res["n"] == 20


def test_a_well_behaved_fit_has_small_influence():
    X, y = _data()
    res = dffitsols(X, y)
    assert max(abs(v) for v in res["dffits"]) < 2.0


def test_an_exactly_fitting_model_gives_nan_not_a_number():
    # zero residual variance means the studentised residual is 0/0. R
    # returns NaN here too; what matters is that it is not silently 0.
    X, y = _data(noise=False)
    res = dffitsols(X, y)
    assert all(v != v for v in res["dffits"])


def test_an_outlier_shows_up_above_the_cutoff():
    X, y = _data()
    y[10] += 25.0
    res = dffitsols(X, y)
    assert abs(res["dffits"][10]) > res["cutoff"]
    assert abs(res["dffits"][10]) == max(abs(v) for v in res["dffits"])


def test_the_cutoff_is_two_root_p_over_n():
    X, y = _data()
    res = dffitsols(X, y)
    assert abs(res["cutoff"] -
               2.0 * math.sqrt(res["p"] / float(res["n"]))) < 1e-9


def test_high_leverage_points_are_at_the_ends():
    X, y = _data()
    lev = dffitsols(X, y)["leverage"]
    assert lev[0] > lev[len(lev) // 2]
    assert lev[-1] > lev[len(lev) // 2]


def test_dffits_is_the_studentised_residual_times_the_leverage_factor():
    X, y = _data()
    y[3] += 4.0
    res = dffitsols(X, y)
    for i in (0, 3, 19):
        h = res["leverage"][i]
        want = res["student"][i] * math.sqrt(h / (1.0 - h))
        assert abs(res["dffits"][i] - want) < 1e-9
