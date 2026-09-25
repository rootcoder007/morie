"""Tests for vitfwd.vit_forward."""

import math

import pytest

from morie.fn import _vitcore as vc
from morie.fn.vitatt import vit_self_attention
from morie.fn.vitcls import vit_cls_token
from morie.fn.vitfwd import vit_forward, vitforward
from morie.fn.vitmlp import vit_mlp_block
from morie.fn.vitptm import vit_patch_embed

IMG = [[1.0, 2.0, 3.0, 4.0],
       [5.0, 6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0, 12.0],
       [13.0, 14.0, 15.0, 16.0]]
P, D, K, DH = 2, 4, 2, 2
N, PDIM, NS = 4, 4, 5          # N = HW/P^2, P^2 C, and N + 1 with the class token
LN_EPS = 1e-6                  # _vitcore's stated epsilon for LN


def _matmul(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def _flat(A):
    return [v for row in A for v in row]


def _ln(v):
    m = sum(v) / len(v)
    q = sum((x - m) ** 2 for x in v) / len(v)
    s = math.sqrt(q + LN_EPS)
    return [(x - m) / s for x in v]


def _z0(w_scale=1.0):
    pe = vit_patch_embed(IMG, P, D, w_scale, 0)
    ct = vit_cls_token(pe["embeddings"], pe["n_patches"], w_scale, pe["skip_used"])
    return pe, ct


def test_vitfwd_zero_layers_is_layernorm_of_the_class_token_row():
    """L = 0 is Eq. (4) applied straight to z_0, the documented degenerate
    anchor: y = LN(z_0^0) and z_L is z_0 untouched."""
    out = vit_forward(IMG, P, D, K, 0)
    pe, ct = _z0()
    assert out["z0"] == ct["z0"]
    assert out["zL"] == ct["z0"]
    assert out["y"] == pytest.approx(_ln(ct["z0"][0]), rel=1e-12)
    assert out["estimate"] == pytest.approx(sum(out["y"]) / D, rel=1e-12)
    assert out["attn"] is None
    assert out["n_patches"] == N and out["seq_len"] == NS
    assert out["patches"] == pe["patches"]
    assert out["skip_used"] == PDIM * D + D + NS * D


def test_vitfwd_layernorm_centres_and_scales_the_representation():
    """LN has gamma = 1, beta = 0, so y sums to zero and its population
    variance is var/(var + eps), within a hair of one."""
    out = vit_forward(IMG, P, D, K, 2)
    y = out["y"]
    assert len(y) == D
    assert sum(y) == pytest.approx(0.0, abs=1e-9)
    assert out["estimate"] == pytest.approx(sum(y) / D, rel=1e-9, abs=1e-12)
    m = sum(y) / D
    v = sum((t - m) ** 2 for t in y) / D
    assert v == pytest.approx(1.0, abs=1e-4)


def test_vitfwd_one_layer_reproduces_equations_2_and_3_from_its_parts():
    """Rebuild z'_1 = MSA(LN(z_0)) + z_0 and z_1 = MLP(LN(z'_1)) + z'_1 out
    of vitatt and vitmlp, in the parameter order the module documents."""
    out = vit_forward(IMG, P, D, K, 1)
    pe, ct = _z0()
    z = [row[:] for row in ct["z0"]]
    skip = ct["skip_used"]

    zn = [_ln(r) for r in z]
    heads, attn = [], None
    for _h in range(K):
        Uq = vc.draw(D, DH, skip, 1.0)
        Uk = vc.draw(D, DH, skip + D * DH, 1.0)
        Uv = vc.draw(D, DH, skip + 2 * D * DH, 1.0)
        skip += 3 * D * DH
        sa = vit_self_attention(_matmul(zn, Uq), _matmul(zn, Uk), _matmul(zn, Uv))
        heads.append(sa["output"])
        attn = sa["attn"]
    cat = [[heads[h][i][c] for h in range(K) for c in range(DH)] for i in range(NS)]
    Umsa = vc.draw(K * DH, D, skip, 1.0)
    skip += K * DH * D
    msa = _matmul(cat, Umsa)
    zp = [[z[i][j] + msa[i][j] for j in range(D)] for i in range(NS)]

    mb = vit_mlp_block([_ln(r) for r in zp], 4 * D, 1.0, skip)
    z1 = [[zp[i][j] + mb["output"][i][j] for j in range(D)] for i in range(NS)]

    assert _flat(out["zL"]) == pytest.approx(_flat(z1), rel=1e-12)
    assert out["y"] == pytest.approx(_ln(z1[0]), rel=1e-12)
    # attn is the last head of the last layer, and it is a softmax
    assert _flat(out["attn"]) == pytest.approx(_flat(attn), rel=1e-12)
    assert out["skip_used"] == mb["skip_used"]


def test_vitfwd_attention_rows_are_probability_distributions():
    out = vit_forward(IMG, P, D, K, 3)
    A = out["attn"]
    assert len(A) == NS and all(len(r) == NS for r in A)
    for row in A:
        assert sum(row) == pytest.approx(1.0, rel=1e-12)
        assert all(0.0 <= a <= 1.0 for a in row)


def test_vitfwd_zero_scale_collapses_every_block_to_the_identity():
    """w_scale = 0 zeroes E, x_class, E_pos and every weight, so z_0 = 0,
    MSA and MLP return zeros, and Eqs. (2)-(3) reduce to z_l = z_{l-1}."""
    for L in (0, 1, 4):
        out = vit_forward(IMG, P, D, K, L, w_scale=0.0)
        assert out["z0"] == [[0.0] * D] * NS
        assert out["zL"] == [[0.0] * D] * NS
        assert out["y"] == [0.0] * D
        assert out["estimate"] == 0.0


def test_vitfwd_stream_accounting_is_exact_in_the_number_of_layers():
    """Per layer: 3 k D D_h for q/k/v, k D_h D for U_msa, 2 D H for the MLP."""
    base = PDIM * D + D + NS * D
    per = 3 * K * D * DH + K * DH * D + 2 * D * (4 * D)
    for L in (0, 1, 2, 5):
        assert vit_forward(IMG, P, D, K, L)["skip_used"] == base + L * per
    assert vit_forward(IMG, P, D, K, 1, mlp_ratio=2)["skip_used"] == (
        base + 3 * K * D * DH + K * DH * D + 2 * D * (2 * D))


def test_vitfwd_reports_its_shapes_and_the_alias_is_the_same_function():
    out = vit_forward(IMG, P, D, K, 2)
    assert out["embed_dim"] == D and out["num_heads"] == K and out["d_head"] == DH
    assert out["num_layers"] == 2 and out["hidden_dim"] == 4 * D
    assert out["n"] == N and out["n_patches"] == N and out["seq_len"] == NS
    assert len(out["z0"]) == NS and all(len(r) == D for r in out["z0"])
    assert vitforward is vit_forward


def test_vitfwd_rejects_bad_input():
    with pytest.raises(ValueError, match="num_heads must divide embed_dim"):
        vit_forward(IMG, P, 4, 3, 1)
    with pytest.raises(ValueError, match="embed_dim must be a positive"):
        vit_forward(IMG, P, 0, 1, 1)
    with pytest.raises(ValueError, match="num_heads must be a positive"):
        vit_forward(IMG, P, D, 0, 1)
    with pytest.raises(ValueError, match="num_layers must be non-negative"):
        vit_forward(IMG, P, D, K, -1)
    with pytest.raises(ValueError, match="mlp_ratio must be a positive"):
        vit_forward(IMG, P, D, K, 1, mlp_ratio=0)
    with pytest.raises(ValueError, match="divide both H and W"):
        vit_forward(IMG, 3, D, K, 1)
