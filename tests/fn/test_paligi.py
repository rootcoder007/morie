"""Tests for paligi (alias of atalib.alibi_position_bias)."""

from morie.fn.atalib import alibi_position_bias
from morie.fn.paligi import head_slopes, paligi, parametric_alibi

Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
V = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_paligi_anchor_press2022_slopes():
    # Press et al. (2022) p.4: for 8 heads the slopes are the printed
    # geometric sequence 1/2^1 .. 1/2^8.
    s = head_slopes(8)
    assert s == [2.0 ** (-(k + 1)) for k in range(8)]
    # for n heads the sequence starts at 2^(-8/n) with that ratio
    s16 = head_slopes(16)
    assert abs(s16[0] - 2.0 ** (-0.5)) < 1e-15
    assert abs(s16[1] / s16[0] - 2.0 ** (-0.5)) < 1e-15


def test_paligi_alias_exact_zero():
    a = paligi(Q=Q, K=K, V=V, slopes=0.25, causal=True)
    b = alibi_position_bias(Q=Q, K=K, V=V, slopes=0.25, causal=True)
    assert a["output"] == b["output"]
    assert a["bias"] == b["bias"]
    assert parametric_alibi is paligi
