"""Tests for wsmpst1.wasserman_posterior_mean (Wasserman 2004, 11.2)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.wsmpst1 import wasserman_posterior_mean


def test_wsmpst1_basic():
    """A Beta(3, 2) posterior on a grid: mean 3/5, sd sqrt(1/25).
    Trapezoid error is O(h^2 f'') ~ 1e-10 at h = 1e-5, below the 1e-8
    tolerance; renormalising by the trapezoid mass cancels the constant."""
    g = [k / 100000 for k in range(100001)]
    dens = [7.3 * t * t * (1 - t) for t in g]      # unnormalised
    r = wasserman_posterior_mean((np.asarray(g), np.asarray(dens)))
    assert r["estimate"] == pytest.approx(0.6, abs=1e-8)
    assert r["posterior_sd"] == pytest.approx(math.sqrt(3 * 2 / (25 * 6)), abs=1e-8)


def test_wsmpst1_edge():
    """A point-symmetric posterior has its mean at the centre."""
    g = [-1 + k / 500 for k in range(1001)]
    r = wasserman_posterior_mean((np.asarray(g), np.asarray([math.exp(-t * t) for t in g])))
    assert r["estimate"] == pytest.approx(0.0, abs=1e-12)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmpst1 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
