"""Tests for gin.gin."""

import pytest

from morie.fn.gin import gin


def test_gin_basic():
    """h_v <- (1 + eps) h_v + sum of neighbour features, recomputed here."""
    A = [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
    H = [[1.0, -1.0], [2.0, 0.5], [0.0, 3.0], [4.0, 1.0]]
    r = gin(A, H, 0.25)
    for v in range(4):
        for j in range(2):
            want = 1.25 * H[v][j] + sum(A[v][u] * H[u][j] for u in range(4))
            assert r["H"][v][j] == pytest.approx(want, rel=1e-15)


def test_gin_edge():
    """eps separates the centre node from its neighbours; A must be n x n."""
    A = [[0, 1], [1, 0]]
    H = [[1.0], [1.0]]
    assert [row[0] for row in gin(A, H, 0.0)["H"]] == [2.0, 2.0]
    with pytest.raises(ValueError, match="n x n"):
        gin([[0, 1, 0]], H)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import importlib as _importlib

# the package also exports a function of this name, so the import
# statement would bind that function, not the module
_doctest_module = _importlib.import_module("morie.fn.gin")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
