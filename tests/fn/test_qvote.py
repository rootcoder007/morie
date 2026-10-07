"""Tests for morie.fn.qvote — Quadratic voting."""

from morie.fn import _array_core as np
from morie.fn.qvote import qvote


def test_qvote_basic():
    I_ = np.array([[10, -5, 1], [1, 10, -5]])
    r = qvote(I_)
    assert "outcomes" in r.value
    assert len(r.value["outcomes"]) == 3


def test_qvote_unanimous():
    I_ = np.array([[10, 10], [10, 10]])
    r = qvote(I_)
    assert all(r.value["outcomes"] == 1)


def test_qvote_budget():
    I_ = np.array([[1, -1]])
    r = qvote(I_, budget=50.0)
    assert r.extra["budget"] == 50.0
