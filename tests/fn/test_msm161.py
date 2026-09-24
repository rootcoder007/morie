"""Verification tests for msm161.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 9, eqs. 9.1 and 9.2 p.339, the hyperplane. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm161 import hyperpl


def test_the_hyperplane_value_is_the_affine_form_of_the_point():
    # eq 9.1: beta_0 + beta_1 x_1 + ... + beta_p x_p
    res = hyperpl([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]], -1.0, [1.0, 1.0])
    assert list(res["value"]) == pytest.approx([1.0, -1.0, 0.0],
                                                abs=1e-12)


def test_the_sign_of_that_form_says_which_half_space_the_point_is_in():
    res = hyperpl([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]], -1.0, [1.0, 1.0])
    assert list(res["side"]) == [1, -1, 0]
    assert list(res["above"]) == [True, False, False]
    assert list(res["below"]) == [False, True, False]
    assert list(res["on_plane"]) == [False, False, True]


def test_the_distance_to_the_plane_is_the_value_over_the_coefficient_norm():
    res = hyperpl([[1.0, 1.0], [0.0, 0.0]], -1.0, [1.0, 1.0])
    nb = math.sqrt(2.0)
    assert res["norm_beta"] == pytest.approx(nb, rel=1e-12)
    assert list(res["distance"]) == pytest.approx([1.0 / nb, 1.0 / nb],
                                                  rel=1e-12)


def test_a_point_on_the_plane_is_at_distance_zero():
    res = hyperpl([[0.5, 0.5]], -1.0, [1.0, 1.0])
    assert res["distance"][0] == pytest.approx(0.0, abs=1e-12)
