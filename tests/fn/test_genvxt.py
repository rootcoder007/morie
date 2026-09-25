"""Tests for genvxt.generalizability_theory."""

import pytest

from morie.fn.genvxt import generalizability_theory


X = [[2, 4, 5], [3, 3, 4], [5, 6, 7], [1, 2, 4], [4, 4, 6]]


def _var(v):
    m = sum(v) / len(v)
    return sum((t - m) ** 2 for t in v) / (len(v) - 1)


def test_genvxt_basic():
    """E rho^2 of the p x i design is coefficient alpha, computed here."""
    r = generalizability_theory(X)
    k = 3
    items = [[row[j] for row in X] for j in range(k)]
    alpha = k / (k - 1) * (1 - sum(_var(c) for c in items) / _var([sum(row) for row in X]))
    assert r["e_rho2"] == pytest.approx(alpha, rel=1e-12)
    assert r["phi"] == pytest.approx(
        r["var_p"] / (r["var_p"] + (r["var_i"] + r["var_pi"]) / k), rel=1e-12)
    assert (r["var_p"], r["var_i"], r["var_pi"]) == pytest.approx((37 / 20, 71 / 60, 17 / 60), rel=1e-12)


def test_genvxt_edge():
    """A D study with more items raises E rho^2 by the Spearman-Brown rule."""
    r = generalizability_theory(X, facets=6)
    assert r["n_i"] == 6
    assert r["e_rho2"] == pytest.approx(r["var_p"] / (r["var_p"] + r["var_pi"] / 6), rel=1e-12)
    with pytest.raises(ValueError, match="two persons"):
        generalizability_theory([[1, 2, 3]])
    with pytest.raises(ValueError, match="at least 1"):
        generalizability_theory(X, facets=0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.genvxt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
