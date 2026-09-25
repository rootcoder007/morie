"""Tests for kmpens.kamath_prompt_ensemble."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.kmpens import kamath_prompt_ensemble

Q = [[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]],
     [[0.5, 0.4, 0.1], [0.2, 0.2, 0.6], [0.05, 0.9, 0.05]]]


def test_kmpens_basic():
    """Weighted arithmetic mean of the prompts' probabilities, recomputed,
    and the prediction is its arg-max."""
    w = [3.0, 1.0]
    result = kamath_prompt_ensemble(Q, weights=w)
    assert isinstance(result, dict)
    got = np.asarray(result["proba"]).tolist()
    for i in range(3):
        p = [(3 * Q[0][i][c] + Q[1][i][c]) / 4 for c in range(3)]
        assert got[i] == pytest.approx(p, rel=1e-14)
        assert int(result["prediction"][i]) == p.index(max(p))


def test_kmpens_edge():
    """'logmean' is the renormalised weighted geometric mean, so a prompt
    giving a class near-zero probability vetoes it."""
    r = kamath_prompt_ensemble(Q, method="logmean")
    got = np.asarray(r["proba"]).tolist()
    for i in range(3):
        g = [math.sqrt(Q[0][i][c] * Q[1][i][c]) for c in range(3)]
        assert got[i] == pytest.approx([v / sum(g) for v in g], rel=1e-12)
    with pytest.raises(ValueError):
        kamath_prompt_ensemble(Q, method="median")


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.kmpens as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
