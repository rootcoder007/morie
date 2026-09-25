"""Tests for getsorg.getis_ord_g (global G, Getis & Ord 1992)."""

import itertools
import math

import pytest

from morie.fn.getsorg import getis_ord_g

X = [3.0, 1.0, 4.0, 1.5, 5.0]
W = [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]]


def _g(x):
    num = sum(W[i][j] * x[i] * x[j] for i in range(5) for j in range(5) if i != j)
    den = sum(x[i] * x[j] for i in range(5) for j in range(5) if i != j)
    return num / den


def test_getsorg_basic():
    """G from its definition, and E[G] and var(G) against the exact
    randomisation distribution: all 5! = 120 relabellings of x."""
    result = getis_ord_g(X, W)
    assert isinstance(result, dict)
    assert result["estimate"] == pytest.approx(_g(X), rel=1e-14)
    gs = [_g(list(p)) for p in itertools.permutations(X)]
    mean = math.fsum(gs) / len(gs)
    var = math.fsum((g - mean) ** 2 for g in gs) / len(gs)
    assert result["expected"] == pytest.approx(mean, rel=1e-12)
    assert result["var"] == pytest.approx(var, rel=1e-10)
    assert result["statistic"] == pytest.approx((_g(X) - mean) / math.sqrt(var), rel=1e-9)


def test_getsorg_edge():
    """E[G] = S0 / (n (n - 1)) = 10 / 20; a diagonal in W is ignored;
    a non-square W is refused."""
    Wd = [row[:] for row in W]
    for i in range(5):
        Wd[i][i] = 7.0
    r = getis_ord_g(X, Wd)
    assert r["expected"] == pytest.approx(0.5, rel=1e-15)
    assert r["estimate"] == pytest.approx(_g(X), rel=1e-14)
    with pytest.raises(ValueError):
        getis_ord_g(X, [row[:4] for row in W])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.getsorg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
