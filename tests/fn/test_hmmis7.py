"""Tests for hmmis7.geron_mistral7b (Mistral 7B architecture accounting)."""

import pytest

from morie.fn.hmmis7 import geron_mistral7b


def test_hmmis7_basic():
    """GQA attention 2 d^2 + 2 d (d / 4) per layer, SwiGLU 3 d d_ff, two
    RMSNorms; total adds untied embedding and head (2 V d) and a final
    norm = 7,241,732,096.  The KV cache is 4x smaller than MHA's
    (32 / 8 heads); the effective span is window x layers."""
    d, L, ff, V = 4096, 32, 14336, 32000
    r = geron_mistral7b([1, 2, 3, 4], n_tokens=2)
    per = 2 * d * d + 2 * d * (d // 4) + 3 * d * ff + 2 * d
    assert r["parameters_per_layer"] == per == 218112000
    assert r["total_parameters"] == L * per + 2 * V * d + d == 7241732096
    assert r["breakdown"]["attention"] == L * (2 * d * d + 2 * d * (d // 4))
    assert r["breakdown"]["feedforward"] == L * 3 * d * ff
    assert r["kv_cache_bytes_mha"] == 2 * L * 6 * d * 2
    assert r["kv_cache_saving"] == pytest.approx(4.0, rel=1e-15)
    assert r["effective_context"] == 4096 * 32
    w = geron_mistral7b([1, 2, 3, 4], n_tokens=0, window=2)
    assert [[int(v) for v in row] for row in w["attention_mask"]] == \
        [[int(i - 2 < j <= i) for j in range(4)] for i in range(4)]


def test_hmmis7_edge():
    """Non-integer token ids raise."""
    with pytest.raises(ValueError):
        geron_mistral7b([0.5, 1.5], 1)
