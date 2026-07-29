"""Tests for algrdy.alammar_greedy_decoding."""

from morie.fn.algrdy import alammar_greedy_decoding


def test_algrdy_basic():
    out = alammar_greedy_decoding([[0.1, 2.0, 0.3]])
    assert out["tokens"] == [1]


def test_algrdy_edge():
    out = alammar_greedy_decoding([[1.0, 1.0]])
    assert out["had_ties"] == [True]
    assert out["tokens"] == [0]
