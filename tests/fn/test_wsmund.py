"""Tests for wsmund.wasserman_undirected_graph (Wasserman 2004, ch. 18)."""

import itertools

import pytest

from morie.fn.wsmund import wasserman_undirected_graph


def _pots():
    agree = lambda t: 2.0 if t[0] == t[1] else 1.0
    tri = lambda t: 3.0 if sum(t) == 3 else 1.0
    return (3, [(0, 1), (0, 1, 2)]), [agree, tri]


def test_wsmund_basic():
    """Z = sum over all 2^n configurations of prod_C psi_C(x_C); every
    probability is that product over Z, enumerated here directly."""
    g, psi = _pots()
    r = wasserman_undirected_graph(g, psi)
    w = {x: psi[0]((x[0], x[1])) * psi[1](x) for x in itertools.product((0, 1), repeat=3)}
    Z = sum(w.values())
    assert r["estimate"] == pytest.approx(Z, abs=1e-12)
    # configuration index is the binary number with x_0 most significant,
    # which is exactly itertools.product order
    for pr, x in zip(r["probabilities"], itertools.product((0, 1), repeat=3)):
        assert pr == pytest.approx(w[x] / Z, abs=1e-15)
    assert r["n_nodes"] == 3 and r["n_cliques"] == 2


def test_wsmund_edge():
    """A clique naming a node outside the graph raises; one potential
    per clique is required."""
    with pytest.raises(ValueError):
        wasserman_undirected_graph((2, [(0, 5)]), [lambda t: 1.0])
    with pytest.raises((ValueError, TypeError)):
        wasserman_undirected_graph((2, [(0, 1)]), [])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmund as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
