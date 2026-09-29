"""landfrag: circle benchmark and fragmentation classes on hand-built maps."""

import math

import pytest

from morie.fn.landfrag import dissection_index, forest_fragmentation


def test_dissection_index():
    r = 3.0
    assert dissection_index(math.pi * r * r, 2 * math.pi * r) == pytest.approx(1.0, rel=1e-15)
    assert dissection_index([4.0, 9.0], [8.0, 12.0]) == pytest.approx(
        [8 / (2 * math.sqrt(4 * math.pi)), 12 / (2 * math.sqrt(9 * math.pi))]
    )


def test_fragmentation_classes():
    g = [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
    r = forest_fragmentation(g)
    # the corner cell's 2x2 window is all forest
    assert r.classes[0][0] == "interior"
    # around the hole: Pf = 8/9, adjacent pairs with forest = 12, both forest = 10
    assert r.pf[1][1] == pytest.approx(8 / 9) and r.pff[1][1] == pytest.approx(10 / 12)
    assert r.classes[1][1] == "perforated"
    # 5 of 9 cells forest next to the hole and the margin: transitional
    assert r.classes[1][3] == "transitional"
    # a straight forest edge: Pf = 6/9 < Pff = 7/10
    e = forest_fragmentation([[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert e.pf[2][1] == pytest.approx(6 / 9) and e.pff[2][1] == pytest.approx(0.7) and e.classes[2][1] == "edge"
    assert r.classes[4][4] is None and r.classes[2][2] is None
    lone = forest_fragmentation([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert lone.classes[1][1] == "patch"
