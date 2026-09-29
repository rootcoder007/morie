"""Tests for barte (alias of hmbart.geron_bart)."""

from morie.fn.barte import bart, barte
from morie.fn.hmbart import geron_bart

SRC = ["the", "cat", "sat", "on", "the", "mat"]


def test_barte_anchor_infilling():
    # Lewis et al. (2020) Sec 2.2: each corrupted span is replaced by a
    # SINGLE <mask> token, so the corrupted length obeys
    # len(corrupted) = len(src) - n_masked + n_spans, and the corrupted
    # sequence is never longer than the source.
    r = barte(SRC, SRC, mask_ratio=0.5, mean_span=2.0, seed=1)
    assert len(r["corrupted"]) == len(SRC) - r["n_masked"] + r["n_spans"]
    assert len(r["corrupted"]) <= len(SRC)
    assert "<mask>" in list(r["corrupted"])
    assert r["n_spans"] <= r["n_masked"]


def test_barte_alias_exact_zero():
    a = barte(SRC, SRC, mask_ratio=0.3, mean_span=3.0, seed=7)
    b = geron_bart(SRC, SRC, mask_ratio=0.3, mean_span=3.0, seed=7)
    assert list(a["corrupted"]) == list(b["corrupted"])
    assert a["estimate"] == b["estimate"]
    assert a["loss"] == b["loss"]
    assert bart is barte


def test_barte_delegates_and_scores_with_the_bigram_model():
    """barte is geron_bart; the default score is the add-one bigram
    cross-entropy of the target, recomputed here."""
    import math

    import pytest

    src = ["the", "cat", "sat", "on", "the", "mat", "today", "ok"]
    tgt = ["a", "b", "a", "b", "c"]
    r = barte(src, tgt, mask_ratio=0.25, seed=2)
    ref = geron_bart(src, tgt, mask_ratio=0.25, seed=2)
    assert r["corrupted"] == ref["corrupted"] and r["loss"] == ref["loss"]
    vocab = sorted(set(tgt))
    V = len(vocab)
    counts = {(p, c): 1.0 for p in vocab + ["<s>"] for c in vocab}
    for i, tok in enumerate(tgt):
        counts[("<s>" if i == 0 else tgt[i - 1], tok)] += 1.0
    lp = []
    for i, tok in enumerate(tgt):
        prev = "<s>" if i == 0 else tgt[i - 1]
        tot = sum(counts[(prev, c)] for c in vocab)
        lp.append(math.log(counts[(prev, tok)] / tot))
    assert r["loss"] == pytest.approx(-sum(lp) / len(lp), rel=1e-13)
    assert V == 3
