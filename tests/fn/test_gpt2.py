"""Tests for gpt2 (alias of hmgpt2.geron_gpt2)."""

from morie.fn.gpt2 import gpt2, gpt_decoder
from morie.fn.hmgpt2 import geron_gpt2

X = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]


def test_gpt2_anchor_radford2019_constants():
    # Radford et al. (2019) Sec 2.3 / Table 2: BPE vocab 50257, context
    # 1024; released sizes (depth, width) = (12,768),(24,1024),
    # (36,1280),(48,1600).
    r = gpt2(X, n_layers=1, n_heads=1, size="small")
    cfg = r["config"]
    assert cfg["vocab_size"] == 50257
    assert cfg["max_len"] == 1024
    sizes = r["all_sizes"]
    # published parameter counts (117M/1542M in Table 2 were later
    # corrected by OpenAI to 124M/1.5B; the exact counts below follow
    # from vocab 50257, context 1024 and the four (depth, width) pairs)
    assert sizes["small"] == 124439808
    assert sizes["xl"] == 1557611200
    assert sizes["small"] < sizes["medium"] < sizes["large"] < sizes["xl"]
    # embedding table dominates the small model (Radford 2019 discussion)
    assert r["embedding_params"] > r["non_embedding_params"]


def test_gpt2_alias_exact_zero():
    a = gpt2(X, n_layers=1, n_heads=1, size="small")
    b = geron_gpt2(X, n_layers=1, n_heads=1, size="small")
    assert a["estimate"] == b["estimate"]
    assert a["total_params"] == b["total_params"]
    assert a["non_embedding_params"] == b["non_embedding_params"]
    assert gpt_decoder is gpt2
