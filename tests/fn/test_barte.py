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
