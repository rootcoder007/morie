"""Tests for wpiece (alias of hmwpt.geron_wordpiece_tokenizer)."""

from morie.fn.hmwpt import geron_wordpiece_tokenizer
from morie.fn.wpiece import wordpiece, wpiece


def test_wpiece_anchor_likelihood_merge():
    # WordPiece merge criterion (Wu et al. 2016 Sec 4.1 restatement):
    # score(A,B) = freq(AB)/(freq(A) freq(B)). In the corpus
    # "hug hug hugs pug pun", the pair (##u, ##g) has freq 4 but its
    # parts are common; (h, ##u) freq 3 with h freq 3. Roundtrip must
    # reconstruct the word.
    r = wpiece(["hug hug hugs pug pun"], vocab_size=14)
    toks = r["tokenize"]("hugs")
    assert "".join(t.replace("##", "") for t in toks) == "hugs"
    assert toks[0][0] == "h"  # first piece is not a continuation
    unk = r["tokenize"]("xyz")
    assert unk == ["[UNK]"]


def test_wpiece_alias_exact_zero():
    a = wpiece(["hug hug hugs pug pun"], vocab_size=14)
    b = geron_wordpiece_tokenizer(["hug hug hugs pug pun"], vocab_size=14)
    assert a["vocab"] == b["vocab"]
    assert a["merges"] == b["merges"]
    assert a["scores"] == b["scores"]
    assert wordpiece is wpiece
