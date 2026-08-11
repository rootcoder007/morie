"""Tests for flsh2 (alias of hmfa.geron_flash_attention)."""

from morie.fn.flsh2 import flash_attention, flsh2
from morie.fn.hmfa import geron_flash_attention

Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
V = [[1.0], [2.0], [3.0], [4.0]]


def test_flsh2_anchor_exactness():
    # Dao et al. (2022) Algorithm 1 is EXACT: tiled equals direct.
    # Independent anchor: a zero query attends uniformly -> mean(V).
    r = flsh2([[0.0]], [[1.0], [3.0]], [[1.0], [3.0]])
    assert abs(r["output"][0][0] - 2.0) < 1e-12
    assert r["max_abs_error"] < 1e-12
    # tiling invariance: block 1 vs block 4 identical outputs
    a = flsh2(Q, K, V, block_size=1)
    b = flsh2(Q, K, V, block_size=4)
    assert max(abs(p[0] - q[0]) for p, q in zip(a["output"], b["output"])) < 1e-12


def test_flsh2_alias_exact_zero():
    a = flsh2(Q, K, V, block_size=2, causal=True)
    b = geron_flash_attention(Q, K, V, block_size=2, causal=True)
    assert a["output"] == b["output"]
    assert a["row_max"] == b["row_max"] and a["row_sum"] == b["row_sum"]
    assert flash_attention is flsh2
