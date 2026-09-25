"""Tests for grnmfo.geron_nmf_objective."""

import pytest

from morie.fn.grnmfo import geron_nmf_objective

X = [[1.0, 0.5, 2.0], [0.0, 1.5, 1.0], [3.0, 0.2, 0.7], [0.4, 0.4, 0.4]]
W = [[0.5, 1.0], [0.2, 0.8], [1.5, 0.1], [0.3, 0.3]]
H = [[1.8, 0.1, 0.6], [0.2, 0.9, 1.1]]


def test_grnmfo_basic():
    """||X - WH||_F^2 with WH, the residual and the relative error
    ||X - WH||_F / ||X||_F recomputed entry by entry."""
    result = geron_nmf_objective(X, W, H)
    assert isinstance(result, dict)
    WH = [[sum(W[i][t] * H[t][j] for t in range(2)) for j in range(3)] for i in range(4)]
    R = [[X[i][j] - WH[i][j] for j in range(3)] for i in range(4)]
    obj = sum(v * v for row in R for v in row)
    assert result["objective"] == pytest.approx(obj, rel=1e-14)
    assert result["relative_error"] == pytest.approx(
        (obj / sum(v * v for row in X for v in row)) ** 0.5, rel=1e-14)
    assert result["rank"] == 2


def test_grnmfo_edge():
    """Negative data is not NMF; mismatched inner dimensions are refused."""
    with pytest.raises(ValueError):
        geron_nmf_objective([[-1.0, 0.0]], [[1.0]], [[1.0, 1.0]])
    with pytest.raises(ValueError):
        geron_nmf_objective(X, W, [[1.0, 1.0, 1.0]])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grnmfo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
