"""Verification tests for km083.

Kamath, Keenan, Somers and Sorenson (2024), the CEAT random-effects bias measure. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km083 import kamath_ch6_ceat_random_effects


def test_ceat_is_the_variance_weighted_mean_of_the_per_sample_effects():
    # CEAT = sum_i v_i WEAT_i / sum_i v_i
    A1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    A2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    W1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    W2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    v = [1.0, 3.0]
    res = kamath_ch6_ceat_random_effects(A1, A2, W1, W2, v)
    weat = list(res["weat"])
    expected = sum(w * e for w, e in zip(v, weat)) / sum(v)
    assert res["estimate"] == pytest.approx(expected, rel=1e-12)
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)


def test_equal_weights_give_the_plain_mean_of_the_effects():
    A1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    A2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    W1 = [[[1.0, 0.0]], [[1.0, 0.0]]]
    W2 = [[[0.0, 1.0]], [[0.0, 1.0]]]
    res = kamath_ch6_ceat_random_effects(A1, A2, W1, W2, [2.0, 2.0])
    weat = list(res["weat"])
    assert res["estimate"] == pytest.approx(sum(weat) / len(weat), rel=1e-12)
