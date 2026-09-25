"""Tests for sentpc.sentencepiece (unigram-LM Viterbi segmentation, Kudo 2018)."""

import math

import pytest

from morie.fn.sentpc import sentencepiece

SP = "▁"
LOGP = {SP: -3.0, SP + "un": -2.0, "un": -2.5, "u": -4.0, "n": -4.0, "do": -2.2,
        "d": -4.5, "o": -4.5, SP + "undo": -5.0, "undo": -4.8, SP + "u": -3.9}


def _best(s):
    # exhaustive search over every split of s into vocabulary pieces
    if not s:
        return 0.0, []
    best = (-math.inf, None)
    for i in range(1, len(s) + 1):
        p = s[:i]
        if p in LOGP:
            lp, rest = _best(s[i:])
            if rest is not None and LOGP[p] + lp > best[0]:
                best = (LOGP[p] + lp, [p] + rest)
    return best


def test_sentpc_basic():
    """Viterbi returns the maximum-log-probability split, checked
    against exhaustive search; spaces become U+2581 and a prefix is
    added."""
    r = sentencepiece("undo", LOGP)
    lp, pieces = _best(SP + "undo")
    assert r["pieces"] == pieces and r["logp"] == pytest.approx(lp, rel=1e-15)
    assert r["pieces"] == [SP + "un", "do"]


def test_sentpc_edge():
    """Without the prefix the leading piece cannot carry U+2581."""
    r = sentencepiece("undo", LOGP, add_prefix=False)
    lp, pieces = _best("undo")
    assert r["pieces"] == pieces == ["un", "do"]
    assert r["logp"] == pytest.approx(-4.7, rel=1e-15)
