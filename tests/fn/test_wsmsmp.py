"""Tests for wsmsmp.wasserman_smoothing_spline."""

import pytest

from morie.fn.wsmsmp import wasserman_smoothing_spline


X = [0.0, 0.5, 1.5, 2.0, 3.5, 4.0, 5.5]
Y = [1.0, 1.8, 1.1, 2.9, 2.2, 3.8, 3.1]


def _D(x):
    n = len(x)
    D = [[0.0] * n for _ in range(n - 2)]
    for i in range(n - 2):
        h1, h2 = x[i + 1] - x[i], x[i + 2] - x[i + 1]
        D[i][i], D[i][i + 1], D[i][i + 2] = 2 / (h1 * (h1 + h2)), -2 / (h1 * h2), 2 / (h2 * (h1 + h2))
    return D


def test_wsmsmp_basic():
    """The fit solves the penalised normal equations (I + lam D'D) m = y,
    with D the divided second difference on the uneven design."""
    lam = 0.7
    m = wasserman_smoothing_spline(X, Y, lam)["estimate"]
    D = _D(X)
    n = len(X)
    for i in range(n):
        dtdm = sum(D[r][i] * sum(D[r][k] * m[k] for k in range(n)) for r in range(n - 2))
        assert m[i] + lam * dtdm == pytest.approx(Y[i], rel=1e-10)


def test_wsmsmp_edge():
    """lam = 0 interpolates; a huge lam gives the least-squares line; a line
    is kept for any lam; 2 < edf < n in between."""
    assert wasserman_smoothing_spline(X, Y, 0.0)["estimate"] == pytest.approx(Y, rel=1e-12)
    xb, yb = sum(X) / 7, sum(Y) / 7
    b = sum((a - xb) * (c - yb) for a, c in zip(X, Y)) / sum((a - xb) ** 2 for a in X)
    line = [yb + b * (a - xb) for a in X]
    # the approach to the least-squares line is O(1/lam): the gap shrinks
    # tenfold per decade of lam (round-off takes over beyond about 1e7)
    gap = [max(abs(u - v) for u, v in zip(wasserman_smoothing_spline(X, Y, lam)["estimate"], line))
           for lam in (1e5, 1e6)]
    assert gap[1] < 2e-6 and 9 < gap[0] / gap[1] < 11
    lin = [2 * a - 1 for a in X]
    assert wasserman_smoothing_spline(X, lin, 5.0)["estimate"] == pytest.approx(lin, abs=1e-12)
    edf = wasserman_smoothing_spline(X, Y, 0.7)["effective_df"]
    assert 2 < edf < 7
    with pytest.raises(ValueError, match="increasing"):
        wasserman_smoothing_spline([0.0, 2.0, 1.0], [1.0, 2.0, 3.0], 1.0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmsmp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
