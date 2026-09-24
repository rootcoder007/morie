"""Tests for kmtot.kamath_tree_of_thoughts."""

import math

import pytest

from morie.fn.kmtot import kamath_tree_of_thoughts


def _score_model(state, b):
    """Model from the docstring example: append '1'..'b', score = i."""
    return [(state + str(i), float(i)) for i in range(1, b + 1)]


def test_kmtot_basic():
    """Greedy (beam=1) search reproduces the docstring worked example."""
    out = kamath_tree_of_thoughts("", 2, 2, _score_model)
    assert isinstance(out, dict)
    for key in ("best_state", "estimate", "best_path", "n_expanded"):
        assert key in out
    assert out["best_state"] == "22"
    assert out["estimate"] == 4.0
    assert out["best_path"] == ["2", "22"]
    assert isinstance(out["n_expanded"], int)
    assert out["n_expanded"] >= 1


def test_kmtot_edge():
    """Wider beam, mixed dead-end model, and invalid arguments."""
    # beam=2 with the docstring model expands exactly 3 nodes
    wide = kamath_tree_of_thoughts("", 2, 2, _score_model, beam=2)
    assert isinstance(wide, dict)
    assert wide["n_expanded"] == 3
    assert math.isfinite(wide["estimate"])

    # Model that produces some dead ends but still completes to max_depth.
    # With beam=2 both children are kept, so the dead end is expanded.
    def mixed_dead_end(state, b):
        if state == "x":
            return [("a", 1.0), ("b", 2.0)]
        elif state == "a":
            return []  # dead end
        elif state == "b":
            return [("c", 3.0)]
        elif state == "c":
            return [("d", 4.0)]
        else:
            return []

    out = kamath_tree_of_thoughts("x", 2, 3, mixed_dead_end, beam=2)
    assert isinstance(out, dict)
    for key in ("best_state", "estimate", "best_path", "n_expanded"):
        assert key in out
    # The best path follows the highest scores: b -> c -> d
    assert out["best_path"] == ["b", "c", "d"]
    assert math.isfinite(out["estimate"])
    # At least four nodes were expanded (including the dead end)
    assert out["n_expanded"] >= 4

    # invalid parameters
    with pytest.raises(ValueError):
        kamath_tree_of_thoughts("x", 0, 2, _score_model)
    with pytest.raises(ValueError):
        kamath_tree_of_thoughts("x", 2, 0, _score_model)
    with pytest.raises(ValueError):
        kamath_tree_of_thoughts("x", 2, 2, _score_model, beam=0)
    with pytest.raises(ValueError):
        kamath_tree_of_thoughts("x", 2, 2, "not callable")


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmtot as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
