"""Tests for mistr. Full anchor: ledger/wave3/anchor_nlp_family.py."""
import math
import pytest
from morie.fn import _array_core as np
from morie.fn.mistr import (apply_rope, attention_span,
                            grouped_query_attention, rms_norm,
                            rope_angles, sliding_window_mask, swiglu)

Q = [0.5, -0.3, 0.2, 0.8]
K = [-0.1, 0.6, 0.4, -0.2]


def test_rope_depends_on_the_relative_position_alone():
    """<R_m q, R_n k> = <R_{m-n} q, k>, to machine precision, for every
    pair -- an implementation with the wrong sign or the wrong channel
    pairing still runs and still trains."""
    worst = 0.0
    for m in range(10):
        for n in range(10):
            lhs = sum(apply_rope(Q, m)[c] * apply_rope(K, n)[c]
                      for c in range(4))
            rhs = sum(apply_rope(Q, m - n)[c] * K[c] for c in range(4))
            worst = max(worst, abs(lhs - rhs))
    assert worst < 1e-13


def test_rope_preserves_norm_and_is_identity_at_zero():
    assert math.sqrt(sum(v * v for v in apply_rope(Q, 7))) == \
        pytest.approx(math.sqrt(sum(v * v for v in Q)), abs=1e-14)
    assert apply_rope(Q, 0) == pytest.approx(Q, abs=1e-15)
    assert rope_angles(8)[0] > rope_angles(8)[-1] > 0.0
    with pytest.raises(ValueError):
        rope_angles(5)


def test_the_sliding_window():
    m = sliding_window_mask(6, 3)
    assert m[5] == [False, False, False, True, True, True]
    assert m[0] == [True, False, False, False, False, False]
    # the span grows with depth, which is the point
    assert attention_span(4096, 32) == 131072
    with pytest.raises(ValueError):
        sliding_window_mask(4, 0)


def test_grouped_query_attention_shapes_and_sharing():
    rng = np.random.default_rng(3)
    Qg = [[rng.standard_normal() for _ in range(8)] for _ in range(5)]
    Kg = [[rng.standard_normal() for _ in range(8)] for _ in range(5)]
    Vg = [[rng.standard_normal() for _ in range(8)] for _ in range(5)]
    # head_dim = 2, so MQA needs K and V just 2 wide -- that narrowing
    # IS the cache saving
    Kq = [r[:2] for r in Kg]
    Vq = [r[:2] for r in Vg]
    mha = grouped_query_attention(Qg, Kg, Vg, 4, 4)
    mqa = grouped_query_attention(Qg, Kq, Vq, 4, 1)
    assert len(mha[0]) == len(mqa[0]) == 8
    assert max(abs(mha[i][c] - mqa[i][c]) for i in range(5)
               for c in range(8)) > 1e-6
    with pytest.raises(ValueError):
        grouped_query_attention(Qg, Kg, Vg, 4, 3)
    with pytest.raises(ValueError):
        grouped_query_attention(Qg, Kq, Vq, 4, 4)


def test_rmsnorm_is_scale_invariant_but_not_shift_invariant():
    """Exact at eps = 0; a realistic eps breaks it only slightly."""
    a = rms_norm([2.0, 4.0, 6.0], eps=0.0)
    b = rms_norm([1.0, 2.0, 3.0], eps=0.0)
    assert a == pytest.approx(b, abs=1e-15)
    shifted = rms_norm([2.0, 3.0, 4.0], eps=0.0)
    assert max(abs(shifted[c] - b[c]) for c in range(3)) > 0.01
    eps_gap = max(abs(rms_norm([2.0, 4.0, 6.0])[c]
                      - rms_norm([1.0, 2.0, 3.0])[c]) for c in range(3))
    assert 1e-12 < eps_gap < 1e-4


def test_swiglu_gates():
    W1 = [[1.0, 0.0], [0.0, 1.0]]
    W2 = [[1.0], [1.0]]
    zero_gate = [[0.0, 0.0], [0.0, 0.0]]
    assert abs(swiglu([1.0, 2.0], W1, W2, zero_gate)[0]) < 1e-15
