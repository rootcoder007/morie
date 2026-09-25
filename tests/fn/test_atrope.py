"""Tests for atrope.rotary_position_embedding."""

import math

import pytest

from morie.fn.atrope import rope, rotary_position_embedding

Q = [1.0, 2.0, -0.5, 0.25, 3.0, -1.0]
THETA = [1.0, 0.1, 0.01]


def _rotate(q, m, theta):
    """R_{theta,m} q, written straight from Su et al. (2021)."""
    out = []
    for i in range(len(q) // 2):
        c, s = math.cos(m * theta[i]), math.sin(m * theta[i])
        a, b = q[2 * i], q[2 * i + 1]
        out += [c * a - s * b, s * a + c * b]
    return out


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def test_atrope_basic():
    """Coordinate pairs are rotated by m * theta_i."""
    assert rotary_position_embedding is rope
    res = rope(Q, 10, THETA)
    assert res["n"] == 6
    assert res["m"] == 10.0
    want = _rotate(Q, 10.0, THETA)
    for got, exp in zip(res["q"], want):
        assert abs(got - exp) < 1e-12
    assert abs(res["norm"] - math.sqrt(_dot(want, want))) < 1e-12
    # the rotation is orthogonal, so the norm is unchanged
    assert abs(res["norm"] - math.sqrt(_dot(Q, Q))) < 1e-12
    # position 0 is the identity
    zero = rope(Q, 0, THETA)
    assert zero["q"] == pytest.approx(Q, abs=1e-15)


def test_atrope_edge():
    """The inner product depends only on the relative position m - n."""
    K = [0.4, -1.3, 2.0, 0.0, -0.7, 0.9]
    for shift in (0.0, 3.0, -5.0):
        a = _dot(rope(Q, 7 + shift, THETA)["q"],
                 rope(K, 2 + shift, THETA)["q"])
        b = _dot(rope(Q, 7, THETA)["q"], rope(K, 2, THETA)["q"])
        assert abs(a - b) < 1e-9
    # theta = 0 leaves every pair alone whatever the position
    flat = rope(Q, 1234, [0.0, 0.0, 0.0])
    assert flat["q"] == pytest.approx(Q, abs=1e-15)
    # a quarter turn on a single pair maps (1, 0) to (0, 1)
    quarter = rope([1.0, 0.0], 1, [math.pi / 2])
    assert quarter["q"][0] == pytest.approx(0.0, abs=1e-15)
    assert abs(quarter["q"][1] - 1.0) < 1e-15
    with pytest.raises(ValueError):
        rope([1.0, 2.0, 3.0], 1, [0.5, 0.5])
    with pytest.raises(ValueError):
        rope(Q, 1, [0.5, 0.5])
