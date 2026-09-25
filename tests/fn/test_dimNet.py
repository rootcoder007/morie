"""Tests for dimNet.dimenet."""

import math

import pytest

from morie.fn.dimNet import angle_between, bessel_basis, spherical_harmonic_basis, triplet_count


def test_dimNet_basic():
    """The angle at j, the radial basis sqrt(2/c) sin(n pi d / c) / d and
    the Legendre angular basis, recomputed here."""
    assert angle_between([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(math.pi / 2, rel=1e-15)
    assert angle_between([1, 1, 0], [0, 0, 0], [1, 0, 0]) == pytest.approx(math.pi / 4, rel=1e-14)
    b = bessel_basis(1.3, cutoff=5.0, n_basis=3)
    assert b == pytest.approx([math.sqrt(2 / 5) * math.sin(n * math.pi * 1.3 / 5) / 1.3 for n in (1, 2, 3)], rel=1e-14)
    x = math.cos(0.7)
    assert spherical_harmonic_basis(0.7, n_basis=4) == pytest.approx(
        [1.0, x, (3 * x * x - 1) / 2, (5 * x ** 3 - 3 * x) / 2], rel=1e-14)


def test_dimNet_edge():
    """Messages interact over triplets: a star with three leaves has six
    ordered (k, j, i) paths through its centre and six directed edges."""
    r = triplet_count({0: [1], 1: [0, 2, 3], 2: [1], 3: [1]})
    assert (r["triplets"], r["pairs"]) == (6, 6)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.dimNet as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
