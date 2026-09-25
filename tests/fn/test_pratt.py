"""Tests for pratt.pretrained_attention (Yang et al. 2016 attention, eqs. 5-7, 10-11)."""

import math

import pytest

from morie.fn.pratt import attention_entropy, pretrained_attention


WS = [[[0.2, 0.1], [0.5, -0.3], [-0.1, 0.4]], [[0.3, 0.3], [-0.6, 0.2]]]
Ww, bw, uw = [[0.4, -0.2], [0.1, 0.3]], [0.0, 0.1], [1.0, -0.5]
Wsn, bsn, us = [[-0.3, 0.2], [0.5, 0.1]], [0.05, 0.0], [0.7, 0.2]
Wc, bc = [[1.0, -1.0], [0.2, 0.3], [-0.4, 0.6]], [0.0, 0.1, -0.1]


def _att(H, W, b, u):
    s = [sum(u[o] * math.tanh(b[o] + sum(W[o][j] * h[j] for j in range(2))) for o in range(2)) for h in H]
    e = [math.exp(v) for v in s]
    a = [v / sum(e) for v in e]
    return a, [sum(a[t] * H[t][f] for t in range(len(H))) for f in range(2)]


def test_pratt_basic():
    """u = tanh(W h + b), alpha = softmax(u' u_ctx), s = sum alpha h at
    the word level then the sentence level, and p = softmax(W_c v + b_c)."""
    r = pretrained_attention(WS, Ww, bw, uw, Wsn, bsn, us, Wc, bc)
    sents = []
    for H, got in zip(WS, r["word_attention"]):
        a, s = _att(H, Ww, bw, uw)
        assert got == pytest.approx(a, rel=1e-12)
        sents.append(s)
    a, v = _att(sents, Wsn, bsn, us)
    assert r["sentence_attention"] == pytest.approx(a, rel=1e-12)
    assert r["document_vector"] == pytest.approx(v, rel=1e-12)
    z = [bc[o] + Wc[o][0] * v[0] + Wc[o][1] * v[1] for o in range(3)]
    assert r["probabilities"] == pytest.approx([math.exp(t) / sum(math.exp(w) for w in z) for t in z], rel=1e-12)
    h = attention_entropy([0.25] * 4)
    assert h["entropy"] == pytest.approx(math.log(4), rel=1e-15) and h["concentration"] == pytest.approx(0.0, abs=1e-15)


def test_pratt_edge():
    """A context vector of the wrong width and an empty sentence raise."""
    with pytest.raises(ValueError):
        pretrained_attention(WS, Ww, bw, [1.0], Wsn, bsn, us, Wc, bc)
    with pytest.raises(ValueError):
        pretrained_attention([[]], Ww, bw, uw, Wsn, bsn, us, Wc, bc)
