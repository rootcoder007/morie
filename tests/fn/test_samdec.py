"""Tests for samdec.sam_mask_decoder (SAM mask decoder, Kirillov et al. 2023)."""

import math

import pytest

from morie.fn.samdec import dice_loss, focal_loss, sam_mask_decoder, two_way_block

P = [[0.5, -0.2, 0.1], [0.0, 0.3, -0.4]]
I = [[0.1 * (i + 1), -0.05 * i, 0.2 * ((-1) ** i)] for i in range(6)]   # 2 x 3 grid


def _attend(Q, K, V):
    out = []
    for q in Q:
        sc = [sum(a * b for a, b in zip(q, kk)) / math.sqrt(len(q)) for kk in K]
        m = max(sc)
        e = [math.exp(v - m) for v in sc]
        out.append([sum(e[j] / sum(e) * V[j][a] for j in range(len(V))) for a in range(len(V[0]))])
    return out


def _add(A, B):
    return [[a + b for a, b in zip(r, s)] for r, s in zip(A, B)]


def test_samdec_basic():
    """One two-way block: prompt self-attention, prompt-to-image and
    image-to-prompt cross-attention, each with a residual, recomputed.
    With one block and no upsampling the logits are the output token's
    dot product with each updated image token."""
    P1 = _add(P, _attend(P, P, P))
    P2 = _add(P1, _attend(P1, I, I))
    I2 = _add(I, _attend(I, P2, P2))
    r = two_way_block(P, I)
    for a, b in zip(r["prompt_tokens"], P2):
        assert a == pytest.approx(b, rel=1e-13, abs=1e-15)
    for a, b in zip(r["image_tokens"], I2):
        assert a == pytest.approx(b, rel=1e-13, abs=1e-15)
    d = sam_mask_decoder(P, I, (2, 3), n_blocks=1, upsample_factor=1)
    logits = [[sum(x * y for x, y in zip(P2[0], I2[i * 3 + j])) for j in range(3)] for i in range(2)]
    for a, b in zip(d["logits"], logits):
        assert a == pytest.approx(b, rel=1e-13, abs=1e-15)


def test_samdec_edge():
    """Upsampling by 2 repeats every logit over a 2 x 2 block; focal and
    Dice losses follow their definitions; a token count that does not
    fill the grid is refused."""
    d1 = sam_mask_decoder(P, I, (2, 3), n_blocks=2, upsample_factor=1)
    d2 = sam_mask_decoder(P, I, (2, 3), n_blocks=2, upsample_factor=2)
    assert d2["shape"] == (4, 6)
    assert d2["logits"] == [[d1["logits"][i // 2][j // 2] for j in range(6)] for i in range(4)]
    p, t = [0.9, 0.2, 0.6], [1.0, 0.0, 0.0]
    fl = sum(-(0.25 if y else 0.75) * (1 - (q if y else 1 - q)) ** 2 * math.log(q if y else 1 - q)
             for q, y in zip(p, t)) / 3
    assert focal_loss(p, t)["loss"] == pytest.approx(fl, rel=1e-14)
    assert dice_loss(p, t)["dice"] == pytest.approx(2 * 0.9 / (1.7 + 1.0), rel=1e-15)
    with pytest.raises(ValueError):
        sam_mask_decoder(P, I[:5], (2, 3))
