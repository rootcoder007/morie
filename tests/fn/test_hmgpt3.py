"""Tests for hmgpt3.geron_gpt3 (GPT-3 175B architecture accounting)."""

import pytest

from morie.fn.hmgpt3 import geron_gpt3


def test_hmgpt3_basic():
    """Per layer 12 d^2 + 13 d; total = L x layer + V d (tied head) +
    n_ctx d positions + 2 d final LayerNorm = 174,604,259,328 for the
    published configuration; KV cache 2 L (prompt + new) d bytes; FLOPs
    2N per token."""
    d, L = 12288, 96
    r = geron_gpt3([1, 2, 3], n_tokens=5)
    per = 12 * d * d + 13 * d
    assert r["parameters_per_layer"] == per == 1812099072
    total = L * per + 50257 * d + 2048 * d + 2 * d
    assert r["total_parameters"] == total == 174604259328
    assert r["breakdown"]["token_embedding"] == 50257 * d == 617558016
    assert r["d_head"] == d // 96 == 128
    assert r["kv_cache_bytes"] == 2 * L * 8 * d * 2
    assert r["flops_per_token"] == 2 * total
    t = geron_gpt3([0, 1], n_tokens=1, n_layers=2, d_model=4, n_heads=2, vocab_size=10, n_ctx=8)
    assert (t["parameters_per_layer"], t["total_parameters"]) == (244, 2 * 244 + 40 + 32 + 8)
    assert t["shape_trace"][-1] == ("logits", (1, 10))


def test_hmgpt3_edge():
    """Non-integer ids, ids past the vocabulary and overlong requests raise."""
    for bad in ([0.5, 1.0], [50257]):
        with pytest.raises(ValueError):
            geron_gpt3(bad, 1)
    with pytest.raises(ValueError):
        geron_gpt3([1, 2], 2047)
