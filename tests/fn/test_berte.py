"""Tests for berte. Full anchor: ledger/wave3/anchor_nlp_family.py."""
import pytest
from morie.fn import _array_core as np
from morie.fn.berte import (attention_weights, bert_encoder, layer_norm)

L, D, H = 5, 8, 2


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(4)
    return [[rng.standard_normal() for _ in range(D)] for _ in range(L)]


def test_attention_rows_sum_to_one_and_are_not_causal(X):
    """The encoder is bidirectional -- an encoder that silently applied
    a causal mask still runs and is a DIFFERENT model."""
    w = attention_weights(X, X, H)
    for h in range(H):
        for i in range(L):
            assert sum(w[h][i]) == pytest.approx(1.0, abs=1e-12)
    assert w[0][0][L - 1] > 0.0
    wc = attention_weights(X, X, H, causal=True)
    assert wc[0][0][L - 1] < 1e-12


def test_padding_does_not_leak_into_the_softmax(X):
    """Otherwise a sequence's answer depends on how much padding its
    batch happened to need."""
    pad = [True, True, True, False, False]
    wp = attention_weights(X, X, H, pad_mask=pad)
    assert max(wp[h][i][j] for h in range(H) for i in range(L)
               for j in (3, 4)) < 1e-12
    short = attention_weights(X[:3], X[:3], H)
    for h in range(H):
        for i in range(3):
            for j in range(3):
                assert wp[h][i][j] == pytest.approx(short[h][i][j],
                                                    abs=1e-12)


def test_layer_norm_centres_and_scales():
    ln = layer_norm([1.0, 2.0, 3.0, 4.0])
    assert sum(ln) == pytest.approx(0.0, abs=1e-12)
    assert sum(v * v for v in ln) / 4 == pytest.approx(1.0, abs=1e-6)
    # and unlike RMSNorm it IS shift-invariant
    a = layer_norm([2.0, 3.0, 4.0])
    b = layer_norm([1.0, 2.0, 3.0])
    assert a == pytest.approx(b, abs=1e-12)


def test_the_encoder_stack(X):
    def blk(seed):
        r = np.random.default_rng(seed)

        def M(a, b):
            return [[r.standard_normal() * 0.2 for _ in range(b)]
                    for _ in range(a)]

        return (M(D, D), M(D, D), M(D, D), M(D, D), M(D, 16),
                [0.0] * 16, M(16, D), [0.0] * D)

    r = bert_encoder(X, [blk(1), blk(2)], H)
    assert r["L"] == L and r["d"] == D
    assert len(r["attention"]) == 2
    assert r["bidirectional"]
    with pytest.raises(ValueError):
        attention_weights(X, X, 3)
