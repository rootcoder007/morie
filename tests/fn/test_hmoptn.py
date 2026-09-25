"""Tests for hmoptn.geron_optuna."""

import pytest

from morie.fn.hmoptn import tpe_suggest


TRIALS = [([0.1], 1.0), ([0.9], 5.0), ([0.2], 1.2), ([0.8], 4.0),
          ([0.15], 0.9), ([0.6], 3.1), ([0.45], 2.0), ([0.3], 1.4)]


def test_hmoptn_basic():
    """The good set is the ceil(gamma n) lowest losses; the suggestion lies
    inside the bounds and, with good trials near 0.1-0.2, below 0.5."""
    r = tpe_suggest(TRIALS, [(0.0, 1.0)], gamma=0.25, seed=1)
    assert (r["n_good"], r["n_bad"]) == (2, 6)
    assert 0.0 <= r["suggestion"][0] <= 1.0
    assert r["suggestion"][0] < 0.5
    assert r["best_so_far"] == 0.9


def test_hmoptn_edge():
    """Seeded suggestions repeat; gamma must lie in (0, 1)."""
    a = tpe_suggest(TRIALS, [(0.0, 1.0)], seed=4)["suggestion"]
    b = tpe_suggest(TRIALS, [(0.0, 1.0)], seed=4)["suggestion"]
    assert a == b
    with pytest.raises(ValueError, match="gamma"):
        tpe_suggest(TRIALS, [(0.0, 1.0)], gamma=1.0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmoptn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
