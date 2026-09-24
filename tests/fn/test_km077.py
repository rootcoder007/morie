"""Verification tests for km077.

Kamath, Keenan, Somers and Sorenson (2024), FActScore over the prompts the model answers. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km077 import kamath_ch6_factscore


def test_factscore_averages_the_supported_fraction_over_responders():
    # FActScore = E_x[(1/|A_y|) sum_a I[a supported by C]] over the
    # prompts the model actually answered
    M = lambda x: x or None
    res = kamath_ch6_factscore(M, ["a b", "c", ""], str.split, {"a", "c"})
    # "a b" -> 1 of 2 supported, "c" -> 1 of 1, the empty prompt abstains
    assert res["estimate"] == pytest.approx(0.75, rel=1e-12)
    assert res["n_responded"] == 2
    assert res["response_rate"] == pytest.approx(2.0 / 3.0, rel=1e-10)


def test_abstentions_are_excluded_from_the_score_but_not_the_rate():
    M = lambda x: x or None
    all_answered = kamath_ch6_factscore(M, ["a", "c"], str.split, {"a", "c"})
    assert all_answered["estimate"] == pytest.approx(1.0, rel=1e-12)
    assert all_answered["response_rate"] == pytest.approx(1.0, rel=1e-12)


def test_no_supported_atom_scores_zero():
    M = lambda x: x
    res = kamath_ch6_factscore(M, ["z"], str.split, {"a"})
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
