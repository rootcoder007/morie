"""Tests for blipqf.blip_qformer (re-export of blip2v.qformer_attend)."""

import math

from morie.fn.blip2v import qformer_attend
from morie.fn.blipqf import blip_qformer, queryingtransformer

# two learnable queries over three frozen patch features
QUERIES = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
PATCHES = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# head width 2; WV widens the value to 4
WQ = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
WK = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
WV = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
      [1.0, 1.0, 1.0]]


def _proj(W, x):
    return [sum(W[o][j] * x[j] for j in range(len(x)))
            for o in range(len(W))]


def _attend(Q, F, wq, wk, wv):
    """Scaled dot-product cross-attention, written from the paper."""
    dk = len(wq)
    out, wts = [], []
    for q in Q:
        qq = _proj(wq, q)
        sc = [sum(a * b for a, b in zip(qq, _proj(wk, f))) / math.sqrt(dk)
              for f in F]
        mx = max(sc)
        e = [math.exp(v - mx) for v in sc]
        z = sum(e)
        w = [v / z for v in e]
        vs = [_proj(wv, f) for f in F]
        wts.append(w)
        out.append([sum(w[j] * vs[j][a] for j in range(len(F)))
                    for a in range(len(vs[0]))])
    return out, wts


def test_blipqf_basic():
    """The Q-Former output width is the query count, not the patch count."""
    assert blip_qformer is qformer_attend
    assert queryingtransformer is qformer_attend
    res = blip_qformer(QUERIES, PATCHES, WQ, WK, WV)
    assert res["n_queries"] == 2
    assert res["n_patches"] == 3
    assert res["compression"] == 1.5
    out, wts = _attend(QUERIES, PATCHES, WQ, WK, WV)
    assert len(res["output"]) == 2
    assert all(len(r) == 4 for r in res["output"])
    for got_row, want_row in zip(res["output"], out):
        for got, want in zip(got_row, want_row):
            assert abs(got - want) < 1e-12
    for got_row, want_row in zip(res["weights"], wts):
        assert abs(sum(got_row) - 1.0) < 1e-12
        for got, want in zip(got_row, want_row):
            assert abs(got - want) < 1e-12
    # the fourth value channel sums the other three, so it is 1 everywhere
    for row in res["output"]:
        assert abs(row[3] - 1.0) < 1e-12


def test_blipqf_edge():
    """More patches do not widen the output; equal scores give equal weight."""
    wide = blip_qformer(QUERIES, PATCHES + PATCHES + PATCHES, WQ, WK, WV)
    assert wide["n_patches"] == 9
    assert wide["n_queries"] == 2
    assert len(wide["output"]) == 2
    assert all(len(r) == 4 for r in wide["output"])
    assert wide["compression"] == 4.5
    # a zero query projects to zero scores, so attention is uniform
    flat = blip_qformer([[0.0, 0.0, 0.0]], PATCHES, WQ, WK, WV)
    assert flat["weights"][0] == [1.0 / 3.0] * 3
    mean = [sum(_proj(WV, f)[a] for f in PATCHES) / 3.0 for a in range(4)]
    for got, want in zip(flat["output"][0], mean):
        assert abs(got - want) < 1e-12
    # one patch leaves all the weight on it, so the output is its value
    single = blip_qformer(QUERIES, [PATCHES[2]], WQ, WK, WV)
    assert single["weights"] == [[1.0], [1.0]]
    for row in single["output"]:
        assert row == _proj(WV, PATCHES[2])
