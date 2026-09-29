"""Tests for grbrnn (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.grbrnn as mod


def test_grbrnn_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_bidirectional_merge_modes_recomputed():
    F = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    Bk = [[0.5, 0.1], [0.2, 0.3], [0.4, 0.9]]
    r = mod.geron_bidirectional_rnn(F, Bk, backward_in_reverse_order=True, combine="mean")
    rev = Bk[::-1]
    assert r["h"] == [[(f + b) / 2 for f, b in zip(F[t], rev[t])] for t in range(3)]
    assert mod.geron_bidirectional_rnn(F, Bk)["h"] == [F[t] + Bk[t] for t in range(3)]
