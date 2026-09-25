"""Tests for wsmlgr.wasserman_logistic_regression (Wasserman 2004, 13.7)."""

import math

import pytest

from morie.fn.wsmlgr import wasserman_logistic_regression


def _data(n=40):
    X = [[1.0, math.sin(1.3 * k)] for k in range(n)]
    y = [1 if ((31 * k + 7) % 53 + 0.5) / 53 < 1 / (1 + math.exp(-(0.3 + 1.2 * x[1]))) else 0
         for k, x in enumerate(X)]
    return X, y


def test_wsmlgr_basic():
    """At the MLE the score X'(y - p) vanishes; se = sqrt diag
    (X'WX)^-1; the log-likelihood is sum y log p + (1-y) log(1-p)."""
    X, y = _data()
    r = wasserman_logistic_regression(X, y)
    b = r["beta"]
    p = [1 / (1 + math.exp(-(x[0] * b[0] + x[1] * b[1]))) for x in X]
    for j in range(2):
        assert abs(sum(x[j] * (yi - pi) for x, yi, pi in zip(X, y, p))) < 1e-9
    I = [[sum(x[a] * x[c] * pi * (1 - pi) for x, pi in zip(X, p)) for c in range(2)] for a in range(2)]
    det = I[0][0] * I[1][1] - I[0][1] * I[1][0]
    assert r["se"] == pytest.approx([math.sqrt(I[1][1] / det), math.sqrt(I[0][0] / det)], rel=1e-9)
    ll = sum(yi * math.log(pi) + (1 - yi) * math.log(1 - pi) for yi, pi in zip(y, p))
    assert r["log_likelihood"] == pytest.approx(ll, abs=1e-10)
    assert r["converged"]


def test_wsmlgr_edge():
    """Intercept-only recovers logit(3/4); separation and a non-binary
    response raise."""
    assert wasserman_logistic_regression([[1.0]] * 4, [1, 1, 1, 0])["beta"][0] == pytest.approx(math.log(3), abs=1e-10)
    with pytest.raises(ValueError):
        wasserman_logistic_regression([[1.0, -1.0], [1.0, -2.0], [1.0, 1.0], [1.0, 2.0]], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        wasserman_logistic_regression([[1.0]] * 3, [0, 1, 2])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmlgr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
