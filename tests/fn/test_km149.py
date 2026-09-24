"""Verification tests for km149.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.21, the Flamingo factorised likelihood. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km149 import kamath_ch9_flamingo_factorized


def test_the_factorised_likelihood_is_the_product_of_the_conditionals():
    # Eq 9.21: p(y|x) = prod_l p(y_l | y_<l, x_<=l)
    res = kamath_ch9_flamingo_factorized([0.5, 0.25])
    assert res["estimate"] == pytest.approx(0.125, rel=1e-12)


def test_a_certain_sequence_has_probability_one():
    res = kamath_ch9_flamingo_factorized([1.0, 1.0, 1.0])
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)


def test_a_longer_sequence_cannot_be_more_probable():
    short = kamath_ch9_flamingo_factorized([0.5])["estimate"]
    long_ = kamath_ch9_flamingo_factorized([0.5, 0.5])["estimate"]
    assert long_ <= short
