"""Tests for wsmgrp.wasserman_graphical_model (Wasserman 2004, ch. 17)."""

import itertools

import pytest

from morie.fn.wsmgrp import wasserman_graphical_model


def test_wsmgrp_basic():
    """The mode is the configuration with the largest clique-potential
    product and its probability is that product over Z (enumerated)."""
    agree = lambda t: 2.0 if t[0] == t[1] else 1.0
    bias = lambda t: 1.5 if t[0] == 1 else 1.0
    g = (3, [(0, 1), (1, 2), (2,)])
    psi = [agree, agree, bias]
    w = {x: agree((x[0], x[1])) * agree((x[1], x[2])) * bias((x[2],))
         for x in itertools.product((0, 1), repeat=3)}
    Z = sum(w.values())
    best = max(w, key=w.get)
    r = wasserman_graphical_model(g, psi)
    assert r["mode"] == list(best)
    assert r["estimate"] == pytest.approx(w[best] / Z, abs=1e-15)
    assert r["partition_function"] == pytest.approx(Z, abs=1e-12)


def test_wsmgrp_edge():
    """Uniform potentials make every configuration equally likely."""
    one = lambda t: 1.0
    r = wasserman_graphical_model((2, [(0, 1)]), [one])
    assert r["estimate"] == pytest.approx(0.25, abs=1e-15)
    assert r["partition_function"] == 4.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmgrp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
