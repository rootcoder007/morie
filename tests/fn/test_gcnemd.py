"""Tests for gcnemd.gcn (Kipf & Welling 2017, eq. 7: the layer before
the renormalisation trick)."""

import math

import pytest

from morie.fn.gcnemd import gcn

A = [[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
X = [[1.0, -0.5], [0.2, 0.8], [-1.0, 0.3], [0.5, 0.5], [2.0, -1.0]]
W = [[0.7, -0.2, 1.0], [0.3, 0.9, -0.4]]


def test_gcnemd_basic():
    """relu((I + D^-1/2 A D^-1/2) X W) recomputed; node 4 is isolated and
    keeps only its own features through the identity term."""
    result = gcn(A, X, W)
    assert isinstance(result, dict)
    d = [sum(r) for r in A]
    s = [0.0 if v == 0 else v ** -0.5 for v in d]
    P = [[(1.0 if i == j else 0.0) + s[i] * A[i][j] * s[j] for j in range(5)] for i in range(5)]
    Z = [[sum(P[i][t] * X[t][f] for t in range(5)) for f in range(2)] for i in range(5)]
    Z = [[sum(Z[i][f] * W[f][o] for f in range(2)) for o in range(3)] for i in range(5)]
    for i in range(5):
        assert result["preactivation"][i] == pytest.approx(Z[i], rel=1e-14, abs=1e-15)
        assert result["H"][i] == pytest.approx([max(v, 0.0) for v in Z[i]], rel=1e-14, abs=1e-15)
    assert result["preactivation"][4] == pytest.approx(
        [sum(X[4][f] * W[f][o] for f in range(2)) for o in range(3)], rel=1e-15)


def test_gcnemd_edge():
    """On a single edge the operator is [[1, 1], [1, 1]] (eigenvalues 0
    and 2, the paper's [0, 2] bound): identical outputs for both nodes.
    A feature matrix with the wrong row count is refused."""
    r = gcn([[0, 1], [1, 0]], [[1.0], [2.0]], [[1.0]])
    assert r["preactivation"] == [[3.0], [3.0]]
    with pytest.raises(ValueError):
        gcn(A, X[:3], W)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gcnemd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
