"""Verification tests for km090.

Kamath, Keenan, Somers and Sorenson (2024), the co-occurrence bias score. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km090 import kamath_ch6_co_occurrence_bias


def test_co_occurrence_bias_is_the_log_ratio_of_conditional_shares():
    # score(w) = log[P(w | A_i) / P(w | A_j)]
    res = kamath_ch6_co_occurrence_bias("nurse", ["nurse nurse doctor"], ["nurse doctor doctor"])
    # 2/3 against 1/3
    assert res["estimate"] == pytest.approx(math.log(2.0), rel=1e-12)


def test_a_balanced_word_scores_zero():
    res = kamath_ch6_co_occurrence_bias("nurse", ["nurse doctor"], ["nurse doctor"])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_the_score_changes_sign_when_the_groups_swap():
    a = kamath_ch6_co_occurrence_bias("nurse", ["nurse nurse doctor"], ["nurse doctor doctor"])["estimate"]
    b = kamath_ch6_co_occurrence_bias("nurse", ["nurse doctor doctor"], ["nurse nurse doctor"])["estimate"]
    assert a == pytest.approx(-b, rel=1e-12)
