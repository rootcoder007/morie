"""Tests for hmbart (re-fixtured: doctests are the worked examples)."""

import doctest

import morie.fn.hmbart as mod


def test_hmbart_doctests():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_single_mask_per_span_and_bigram_loss():
    """Every span collapses to one <mask>; the loss is the add-one bigram
    cross-entropy of the target."""
    import math

    import pytest

    src = list("abcdefghijklmnop")
    tgt = ["x", "y", "x", "z", "x"]
    r = mod.geron_bart(src, tgt, mask_ratio=0.3, seed=4)
    assert r["corrupted"].count("<mask>") == r["n_spans"]
    assert len(r["corrupted"]) == len(src) - r["n_masked"] + r["n_spans"]
    vocab = sorted(set(tgt))
    cnt = {(p, c): 1.0 for p in vocab + [None] for c in vocab}
    for i, t in enumerate(tgt):
        cnt[(None if i == 0 else tgt[i - 1], t)] += 1.0
    lp = [
        math.log(
            cnt[(None if i == 0 else tgt[i - 1], t)] / sum(cnt[(None if i == 0 else tgt[i - 1], c)] for c in vocab)
        )
        for i, t in enumerate(tgt)
    ]
    assert r["loss"] == pytest.approx(-sum(lp) / 5, rel=1e-13)
