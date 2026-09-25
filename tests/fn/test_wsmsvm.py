"""Tests for wsmsvm.wasserman_svm (linear SVM by the dual)."""

import math

import pytest

from morie.fn.wsmsvm import wasserman_svm


def test_wsmsvm_basic():
    """Separable data, hard margin: every point satisfies
    y (w'x + b) >= 1, the support vectors sit on the margin, w =
    sum a_i y_i x_i, and sum a_i y_i = 0."""
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 3.5], [-1.0, -0.5], [-2.0, -1.5], [0.0, -1.0]]
    y = [1, 1, 1, -1, -1, -1]
    r = wasserman_svm(X, y)
    w, b, a = r["w"], r["b"], r["alphas"]
    for xi, yi in zip(X, y):
        assert yi * (w[0] * xi[0] + w[1] * xi[1] + b) >= 1 - 1e-8
    for i in r["support_vectors"]:
        assert y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b) == pytest.approx(1.0, abs=1e-8)
    assert w == pytest.approx([sum(a[i] * y[i] * X[i][j] for i in range(6)) for j in range(2)], abs=1e-10)
    assert sum(ai * yi for ai, yi in zip(a, y)) == pytest.approx(0.0, abs=1e-10)
    assert r["estimate"] == pytest.approx(2 / math.hypot(*w), rel=1e-12)


def test_wsmsvm_edge():
    """Two points: w = (0.5, 0.5), b = 0; one class or bad labels raise."""
    r = wasserman_svm([[-1.0, -1.0], [1.0, 1.0]], [-1, 1])
    assert r["w"] == pytest.approx([0.5, 0.5], abs=1e-10)
    assert r["b"] == pytest.approx(0.0, abs=1e-10)
    with pytest.raises(ValueError):
        wasserman_svm([[0.0], [1.0]], [1, 1])
    with pytest.raises(ValueError):
        wasserman_svm([[0.0], [1.0]], [0, 1])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmsvm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
