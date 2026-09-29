"""Tests for km012.kamath_ch2_scaled_dot_attention (re-fixtured from the doctest)."""

import doctest

import morie.fn.km012 as mod


def test_km012_doctest():
    r = doctest.testmod(mod)
    assert r.failed == 0
    assert r.attempted > 0


def test_km012_edge():
    import pytest

    from morie.fn.km012 import kamath_ch2_scaled_dot_attention

    with pytest.raises(ValueError):
        kamath_ch2_scaled_dot_attention([[1.0]], [[1.0]], [[1.0]], d_k=9)


def test_scaled_dot_product_recomputed():
    import math

    import pytest

    Q = [[1.0, 0.0], [0.5, 0.5]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    V = [[1.0], [0.0], [2.0]]
    r = mod.kamath_ch2_scaled_dot_attention(Q, K, V, d_k=2)
    for i, q in enumerate(Q):
        s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(2) for k in K]
        e = [math.exp(v - max(s)) for v in s]
        assert r["output"][i][0] == pytest.approx(sum(e[j] * V[j][0] for j in range(3)) / sum(e), rel=1e-13)
