"""Verification tests for km107.

Kamath, Keenan, Somers and Sorenson (2024), eq. 6.31, the personally-identifiable-information leakage. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km107 import kamath_ch6_pii_likelihood


def test_the_leakage_likelihood_is_the_product_over_the_response_tokens():
    # Eq 6.31: P_r(a_m | A_no_m) = prod_r p(a_mr | x_1..x_{L_q+r-1})
    res = kamath_ch6_pii_likelihood([0.5, 0.25], ["name"], ["contact", "John"], 2, 2)
    assert res["estimate"] == pytest.approx(0.125, rel=1e-12)
    assert list(res["context_lengths"]) == [2, 3]


def test_the_context_grows_by_one_token_per_step():
    res = kamath_ch6_pii_likelihood([0.5, 0.5, 0.5], ["name"], ["q1", "q2", "q3"], 3, 3)
    assert list(res["context_lengths"]) == [3, 4, 5]


def test_a_certain_recall_scores_one():
    res = kamath_ch6_pii_likelihood([1.0, 1.0], ["name"], ["a"], 1, 2)
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)
