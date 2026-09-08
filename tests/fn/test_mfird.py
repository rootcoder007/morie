"""Tests for mfird (MIRT discrimination to factor loading, Reckase 2009).

Replaces the generated stub, which imported ``mirt_factor_loading``.
"""

import math

from morie.fn.mfird import mfird


def test_loadings_and_communalities_agree():
    a = [[1.0, 0.5], [0.8, 0.0], [0.3, 1.2]]
    res = mfird(a)
    for i, row in enumerate(res["loadings"]):
        h2 = sum(v * v for v in row)
        assert abs(res["communalities"][i] - h2) < 1e-12
        assert 0.0 <= h2 < 1.0


def test_a_larger_discrimination_gives_a_larger_loading():
    small = mfird([[0.2, 0.0]])["loadings"][0][0]
    large = mfird([[3.0, 0.0]])["loadings"][0][0]
    assert large > small
    assert large < 1.0                    # a loading cannot reach one


def test_zero_discrimination_gives_zero_loading():
    res = mfird([[0.0, 0.0]])
    assert abs(res["loadings"][0][0]) < 1e-12
    assert abs(res["communalities"][0]) < 1e-12


def test_the_round_trip_returns_the_discriminations():
    a = [[1.0, 0.5], [0.8, 0.2]]
    load = mfird(a)["loadings"]
    # the inverse direction returns discriminations, not loadings
    back = mfird(load, inverse=True)["discriminations"]
    for i in range(2):
        for k in range(2):
            assert abs(back[i][k] - a[i][k]) < 1e-9


def test_intercepts_become_thresholds():
    a = [[1.0, 0.0]]
    res = mfird(a, d=[-0.5])
    assert res["thresholds"] is not None
    assert len(res["thresholds"]) == 1


def test_validation():
    for call in (lambda: mfird([[1.0, 0.5], [1.0]]),
                 lambda: mfird([[1.0, 0.5]], d=[0.1, 0.2]),
                 lambda: mfird([[0.99, 0.99]], inverse=True)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
