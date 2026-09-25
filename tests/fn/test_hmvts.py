"""Tests for hmvts.geron_voting_soft (soft voting across classifiers)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.hmvts import geron_voting_soft

# three members, four rows, three classes
PR = [[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]],
      [[0.5, 0.1, 0.4], [0.3, 0.3, 0.4], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]],
      [[0.1, 0.8, 0.1], [0.6, 0.2, 0.2], [0.34, 0.33, 0.33], [0.2, 0.2, 0.6]]]


def test_hmvts_basic():
    """proba is the weight-normalised average of the members'
    probabilities and the prediction its argmax, recomputed here."""
    w = [1.0, 2.0, 1.0]
    result = geron_voting_soft(PR, weights=w)
    assert isinstance(result, dict)
    ws = [v / sum(w) for v in w]
    got = np.asarray(result["proba"]).tolist()
    for i in range(4):
        p = [sum(ws[m] * PR[m][i][c] for m in range(3)) for c in range(3)]
        assert got[i] == pytest.approx(p, rel=1e-12, abs=0)
        assert int(result["prediction"][i]) == p.index(max(p))


def test_hmvts_edge():
    """Hard voting counts each member's argmax; with y the errors are
    the misclassification rates, counted here."""
    y = [0, 2, 1, 2]
    r = geron_voting_soft(PR, y=y)
    votes = [[PR[m][i].index(max(PR[m][i])) for m in range(3)] for i in range(4)]
    hard = [max(range(3), key=lambda c: (v.count(c), -c)) for v in votes]
    assert [int(v) for v in r["hard_prediction"]] == hard
    soft = [int(v) for v in r["prediction"]]
    assert r["soft_error"] == sum(a != b for a, b in zip(soft, y)) / 4
    assert r["hard_error"] == sum(a != b for a, b in zip(hard, y)) / 4
    with pytest.raises(ValueError):
        geron_voting_soft([[0.5, 0.5]])


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmvts as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
