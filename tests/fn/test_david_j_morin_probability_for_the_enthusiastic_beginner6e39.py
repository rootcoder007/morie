"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner6e39.

Morin (2016), eq (6.39) -- the group average under a linear relation. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner6e39 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39


def test_group_average_of_y_is_the_slope_times_the_group_average_of_x():
    # eq (6.39): with y = m x, the group mean of y is m times that of x
    for m, xavg in ((2.0, 3.0), (-0.5, 10.0), (0.0, 4.0)):
        assert david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39(m, xavg)["yavg"] == pytest.approx(m * xavg, rel=1e-12)


def test_group_average_is_linear_in_the_group_average_of_x():
    m = 1.7
    a = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39(m, 2.0)["yavg"]
    b = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_39(m, 4.0)["yavg"]
    assert b == pytest.approx(2.0 * a, rel=1e-12)
