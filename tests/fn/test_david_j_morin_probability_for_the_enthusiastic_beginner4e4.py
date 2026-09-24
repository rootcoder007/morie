"""Verification tests for david_j_morin_probability_for_the_enthusiastic_beginner4e4.

Morin (2016), eq (4.4) -- probability over a centred interval. Expected values are recomputed
from the identity in the test body.
"""

import math

import pytest

from morie.fn.david_j_morin_probability_for_the_enthusiastic_beginner4e4 import david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4


def test_centred_interval_probability_on_a_uniform_density():
    # eq (4.4): P over [T - dT/2, T + dT/2]; a unit density gives the width
    grid = [i / 200.0 for i in range(201)]
    density = [1.0] * 201
    res = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4(grid, density, 0.5, 0.2)
    assert res["probability"] == pytest.approx(0.2, abs=1e-9)
    assert res["center"] == pytest.approx(0.5, rel=1e-12)
    assert res["width"] == pytest.approx(0.2, rel=1e-12)


def test_centred_interval_probability_grows_with_the_width():
    grid = [i / 200.0 for i in range(201)]
    density = [1.0] * 201
    narrow = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4(grid, density, 0.5, 0.1)["probability"]
    wide = david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_4(grid, density, 0.5, 0.4)["probability"]
    assert wide > narrow
