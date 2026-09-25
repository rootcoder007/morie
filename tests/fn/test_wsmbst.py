"""Tests for wsmbst.wasserman_boosting."""

import math

import pytest

from morie.fn.wsmbst import wasserman_boosting


def test_wsmbst_basic():
    """A single-threshold problem is solved by one perfect stump."""
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [1, 1, -1, -1]
    out = wasserman_boosting(X, y, None, 5)

    assert out["n"] == 4
    assert out["estimate"] == 0.0
    assert out["rounds_used"] == 1
    assert out["prediction"] == y
    # A perfect stump gets the documented capped weight and stops the run.
    assert out["alphas"] == [10.0]

    # Two informative features, one of which separates: still one stump.
    X2 = [[0.0, 10.0], [1.0, 11.0], [5.0, 0.0], [6.0, 1.0]]
    out2 = wasserman_boosting(X2, y, None, 8)
    assert out2["estimate"] == 0.0
    assert out2["rounds_used"] == 1
    assert out2["prediction"] == y
    assert out2["alphas"] == [10.0]


def test_wsmbst_xor_is_beyond_a_single_stump():
    """No axis-aligned stump beats chance on XOR, so boosting stops at
    round 0 and the empty committee misclassifies half the sample."""
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [-1, 1, 1, -1]
    out = wasserman_boosting(X, y, None, 10)

    assert out["rounds_used"] == 0
    assert out["alphas"] == []
    # F is all zeros, so sign(F) is +1 everywhere; two of four labels are -1.
    assert out["prediction"] == [1, 1, 1, 1]
    assert out["estimate"] == pytest.approx(2.0 / 4.0, rel=1e-12)


def test_wsmbst_imperfect_stump_uses_the_adaboost_weight():
    """One label breaks the threshold: the first stump has weighted error
    1/4, so its weight is (1/2) log 3."""
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [1, -1, -1, -1]
    out = wasserman_boosting(X, y, None, 1)

    assert out["rounds_used"] == 1
    # thr = 0 with sign +1 predicts [1, -1, -1, -1] -- in fact perfect here.
    assert out["estimate"] == 0.0

    # Now a genuinely non-separable 1-D problem: alternating labels.
    alt = wasserman_boosting([[0.0], [1.0], [2.0], [3.0]], [1, -1, 1, -1], None, 1)
    assert alt["rounds_used"] == 1
    assert len(alt["alphas"]) == 1
    # The best stump misclassifies exactly one of four equally weighted
    # points, so err = 1/4 and alpha = (1/2) log((1 - err) / err).
    err = 0.25
    assert alt["alphas"][0] == pytest.approx(
        0.5 * math.log((1.0 - err) / err), rel=1e-12
    )
    # One round of a stump that is right on 3 of 4 points.
    assert alt["estimate"] == pytest.approx(1.0 / 4.0, rel=1e-12)


def test_wsmbst_custom_weak_learner():
    """A callable weak learner is used in place of the built-in stumps."""
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [1, 1, -1, -1]

    calls = []

    def factory(Xt, yt, w):
        calls.append(float(sum(float(v) for v in w)))
        # A perfect learner for this problem.
        return lambda Xq: [1.0 if float(row[0]) <= 1.0 else -1.0 for row in Xq]

    out = wasserman_boosting(X, y, factory, 4)
    assert calls and calls[0] == pytest.approx(1.0, rel=1e-12)
    assert out["rounds_used"] == 1
    assert out["estimate"] == 0.0
    assert out["prediction"] == y
    assert out["alphas"] == [10.0]


def test_wsmbst_edge():
    """Label domain, length agreement, and T >= 1 are enforced."""
    X = [[0.0], [1.0], [2.0], [3.0]]
    with pytest.raises(ValueError):
        wasserman_boosting(X, [1, 2, 3, 4], None, 3)
    with pytest.raises(ValueError):
        wasserman_boosting(X, [1, 1, -1], None, 3)
    with pytest.raises(ValueError):
        wasserman_boosting(X, [1, 1, -1, -1], None, 0)
    with pytest.raises(ValueError):
        wasserman_boosting(X, [1, 1, -1, -1], None, -2)

    # More rounds than needed changes nothing once a perfect stump is found.
    a = wasserman_boosting(X, [1, 1, -1, -1], None, 1)
    b = wasserman_boosting(X, [1, 1, -1, -1], None, 50)
    assert a["rounds_used"] == b["rounds_used"] == 1
    assert a["prediction"] == b["prediction"]
    assert a["alphas"] == b["alphas"]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmbst as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
