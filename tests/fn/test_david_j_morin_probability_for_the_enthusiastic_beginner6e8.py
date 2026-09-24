"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner6e8.

Morin (2016), eq (6.8) -- covariance of zero-mean variables. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e8 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8


def test_zero_mean_covariance_is_the_mean_product():
    # eq (6.8): with zero means, Cov(X,Y) = E(XY)
    x = [-2.0, -1.0, 1.0, 2.0]
    y = [-3.0, 1.0, -1.0, 3.0]
    assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8(x, y)["cov"] == pytest.approx(
        sum(a * b for a, b in zip(x, y)) / len(x), rel=1e-12)


def test_zero_mean_form_refuses_data_that_is_not_centred():
    with pytest.raises(ValueError):
        david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_8([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
