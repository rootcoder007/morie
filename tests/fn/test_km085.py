"""Verification tests for km085.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.9, the Categorical Bias Score. Expected values are
recomputed in the test body.
"""

import math
import statistics

import pytest

from morie.fn.km085 import kamath_ch6_cbs_variance


def test_categorical_bias_score_is_the_mean_variance_of_the_log_ratios():
    # Eq 6.9: CBS = (1/|W|) sum_w Var_a log(p_a / p_prior)
    words = ["w1", "w2"]
    attrs = ["a", "b"]
    p_a = [[0.3, 0.7], [0.4, 0.6]]
    p_prior = [[0.5, 0.5], [0.5, 0.5]]
    res = kamath_ch6_cbs_variance(words, attrs, p_a, p_prior)
    per_word = []
    for row, prior in zip(p_a, p_prior):
        logs = [math.log(p / q) for p, q in zip(row, prior)]
        mean = sum(logs) / len(logs)
        per_word.append(sum((v - mean) ** 2 for v in logs) / len(logs))
    assert res["per_word"] == pytest.approx(per_word, rel=1e-12)
    assert res["estimate"] == pytest.approx(
        sum(per_word) / len(per_word), rel=1e-12)


def test_a_word_with_no_bias_contributes_zero_variance():
    res = kamath_ch6_cbs_variance(["w"], ["a", "b"], [[0.5, 0.5]], [[0.5, 0.5]])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_a_more_skewed_word_raises_the_score():
    mild = kamath_ch6_cbs_variance(["w"], ["a", "b"], [[0.45, 0.55]], [[0.5, 0.5]])["estimate"]
    severe = kamath_ch6_cbs_variance(["w"], ["a", "b"], [[0.05, 0.95]], [[0.5, 0.5]])["estimate"]
    assert severe > mild


def test_a_prior_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        kamath_ch6_cbs_variance(["w1", "w2"], ["a", "b"], [[0.3, 0.7], [0.4, 0.6]], [[0.5, 0.5]])
