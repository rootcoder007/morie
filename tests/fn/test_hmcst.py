"""Tests for hmcst.geron_contrastive_learning (InfoNCE, Geron ch. 16)."""

import math

import pytest

from morie.fn.hmcst import geron_contrastive_learning


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(
        sum(x * x for x in a) * sum(y * y for y in b))


def _emb():
    return [[1.0, 0.2, -0.5], [0.9, 0.1, -0.4], [-0.3, 1.0, 0.2],
            [-0.2, 0.8, 0.5], [0.4, -0.6, 1.0]]


def test_hmcst_basic():
    """Per-anchor InfoNCE recomputed from cosine similarities: the
    denominator is the positive plus every other non-self row."""
    E, pos, tau = _emb(), [1, 0, 3, 2, 0], 0.5
    result = geron_contrastive_learning(E, pos, tau=tau)
    assert isinstance(result, dict)
    B = len(E)
    per = []
    for i in range(B):
        num = math.exp(_cos(E[i], E[pos[i]]) / tau)
        den = sum(math.exp(_cos(E[i], E[j]) / tau) for j in range(B) if j != i)
        per.append(-math.log(num / den))
    assert result["per_anchor_loss"] == pytest.approx(per, rel=1e-12, abs=0)
    assert result["loss"] == pytest.approx(sum(per) / B, rel=1e-12, abs=0)
    assert result["n_negatives"] == B - 2


def test_hmcst_edge():
    """A batch of one has no negatives and is refused; an out-of-range
    positive index is refused too."""
    with pytest.raises(ValueError):
        geron_contrastive_learning([[1.0, 0.0]], [0])
    with pytest.raises(ValueError):
        geron_contrastive_learning(_emb(), [1, 0, 3, 2, 9])


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmcst as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
