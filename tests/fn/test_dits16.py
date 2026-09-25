"""Tests for dits16: Scalable Diffusion Models with Transformers.

Peebles and Xie (2023), arXiv:2212.09748. Token counts, the adaLN-Zero
block and the Gflops measure, with Table 4 of the paper as reference.
"""

import math

import pytest

from morie.fn.dits16 import (adaln_zero, dit_block, dit_diffusion_transformer,
                             gflops, patch_grid, scaling_comparison)

# (depth, width) of the DiT configurations, paper Table 1
CFG = {"S": (12, 384), "B": (12, 768), "L": (24, 1024), "XL": (28, 1152)}


def test_dits16_basic():
    """T = (I/p)^2 for a 32x32 latent (256x256 images)."""
    assert patch_grid(32, 2)["tokens"] == 256
    assert patch_grid(32, 4)["tokens"] == 64
    assert patch_grid(32, 8)["tokens"] == 16
    # halving the patch quadruples the tokens
    assert patch_grid(32, 2)["tokens"] == 4 * patch_grid(32, 4)["tokens"]


def test_gflops_reproduces_the_papers_table_4():
    # The formula omits the patch embedding and final layer, which is
    # well under 1% of the cost for these models; the paper's own
    # figures are the reference.
    table = {("B", 2): 23.01, ("L", 2): 80.71, ("XL", 2): 118.64,
             ("B", 4): 5.56, ("L", 4): 19.70}
    for (name, p), paper in table.items():
        L, d = CFG[name]
        g = gflops(patch_grid(32, p)["tokens"], L, d)["gflops"]
        assert g == pytest.approx(paper, rel=0.01), (name, p, g)
    # and the formula itself, recomputed
    T, L, d = 256, 28, 1152
    want = L * (4 * T * d * d + 2 * T * T * d + 8 * T * d * d) / 1e9
    assert gflops(T, L, d)["gflops"] == pytest.approx(want, rel=1e-15)


def test_adaln_zero_is_the_identity_at_initialisation():
    cond, h = [0.3, -1.2], [1.0, 2.0, 4.0]
    Z = [[0.0, 0.0]] * 3
    Ws = [[0.1, 0.2], [0.0, -0.3], [0.5, 0.5]]
    out = adaln_zero(cond, h, Ws, Ws, Z)
    m = sum(h) / 3
    s = math.sqrt(sum((v - m) ** 2 for v in h) / 3 + 1e-6)
    g = [sum(Ws[o][j] * cond[j] for j in range(2)) for o in range(3)]
    for i in range(3):
        assert out["modulated"][i] == pytest.approx(
            (h[i] - m) / s * (1 + g[i]) + g[i], rel=1e-12)
    assert out["gate"] == [0.0, 0.0, 0.0] and out["identity_at_init"]
    # with every gate zero, a whole block passes the input through
    blk = dit_block(h, cond, lambda x: [10.0] * 3, lambda x: [7.0] * 3,
                    Ws, Ws, Z, Ws, Ws, Z)
    assert blk["output"] == h and blk["identity_at_init"]
    assert dit_diffusion_transformer is dit_block


def test_a_nonzero_gate_adds_the_gated_residual():
    cond, h = [1.0], [0.0, 1.0]
    one = [[1.0], [1.0]]
    zero = [[0.0], [0.0]]
    blk = dit_block(h, cond, lambda x: [2.0, 3.0], lambda x: [0.0, 0.0],
                    zero, zero, [[0.5], [0.25]], zero, zero, zero)
    assert blk["output"] == pytest.approx([0.0 + 0.5 * 2.0, 1.0 + 0.25 * 3.0],
                                          rel=1e-15)
    assert not blk["identity_at_init"]
    del one


def test_parameters_match_the_papers_model_sizes():
    ranked = scaling_comparison([(n, 32, 2, *CFG[n]) for n in CFG])["ranked"]
    by = {r["name"]: r for r in ranked}
    paper_params = {"B": 130e6, "L": 458e6, "XL": 675e6}
    for name, want in paper_params.items():
        # embeddings, timestep MLP and head are the missing remainder
        assert by[name]["parameters"] == pytest.approx(want, rel=0.03)
    # ranked by cost, and the token axis moves cost at fixed parameters
    assert [r["name"] for r in ranked] == ["S", "B", "L", "XL"]
    a = scaling_comparison([("p2", 32, 2, 12, 768), ("p4", 32, 4, 12, 768)])
    r2, r4 = {r["name"]: r for r in a["ranked"]}["p2"], {r["name"]: r for r in a["ranked"]}["p4"]
    assert r2["parameters"] == r4["parameters"] and r2["gflops"] > 4 * r4["gflops"] * 0.9


def test_dits16_edge():
    with pytest.raises(ValueError, match="does not divide"):
        patch_grid(32, 3)
    with pytest.raises(ValueError, match="positive"):
        patch_grid(32, 0)
    with pytest.raises(ValueError, match="positive"):
        gflops(0, 12, 768)
    with pytest.raises(ValueError, match="mis-sized"):
        adaln_zero([1.0], [1.0, 2.0], [[0.0]], [[0.0]], [[0.0]])
