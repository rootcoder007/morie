"""Verification tests for km092.

Kamath, Keenan, Somers and Sorenson (2024), the stereotypical-association count. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km092 import kamath_ch6_stereotypical_assoc


def test_stereotypical_association_counts_attribute_mentions():
    # ST(w)_i = sum_a sum_Yhat C(a, Yhat) I(C(w, Yhat) > 0)
    res = kamath_ch6_stereotypical_assoc("nurse", ["she"], ["she is a nurse", "she is a doctor"])
    # only the first output contains the word, and it mentions "she" once
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)
    assert res["n_outputs_with_w"] == 1


def test_repeated_attribute_mentions_are_counted():
    res = kamath_ch6_stereotypical_assoc("nurse", ["she"],
               ["she and she are nurse", "she is a doctor"])
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)


def test_a_word_absent_from_every_output_scores_zero():
    res = kamath_ch6_stereotypical_assoc("pilot", ["she"], ["she is a nurse"])
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
    assert res["n_outputs_with_w"] == 0
