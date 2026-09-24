"""Verification tests for grlof.geron_local_outlier_factor.

LOF is the mean ratio of a neighbour's local reachability density to the
point's own, so a point as dense as its neighbourhood scores about 1 and
a sparse one scores well above 1. The tests assert that defining
behaviour rather than recorded numbers.
"""

import math

import pytest

from morie.fn.grlof import geron_local_outlier_factor


def _grid():
    # a tight 3 by 3 lattice: every interior point is as dense as its
    # neighbours, so every LOF sits near 1
    return [[float(i), float(j)] for i in range(3) for j in range(3)]


def test_a_uniform_lattice_scores_about_one_everywhere():
    res = geron_local_outlier_factor(_grid(), k=3)
    lof = res["lof"]
    for v in lof:
        assert 0.5 < float(v) < 2.0


def test_an_isolated_point_scores_above_its_neighbours():
    pts = _grid() + [[20.0, 20.0]]
    res = geron_local_outlier_factor(pts, k=3)
    lof = [float(v) for v in res["lof"]]
    assert lof[-1] > max(lof[:-1])
    assert lof[-1] > 1.5


def test_every_score_is_finite_and_positive():
    res = geron_local_outlier_factor(_grid(), k=2)
    lof = [float(v) for v in res["lof"]]
    assert all(math.isfinite(v) and v > 0.0 for v in lof)


def test_k_cannot_reach_the_sample_size():
    # a point cannot be its own neighbour, so k is at most n - 1
    with pytest.raises(ValueError):
        geron_local_outlier_factor(_grid(), k=9)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grlof as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
