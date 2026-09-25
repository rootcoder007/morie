"""Tests for cvxstx.boyd_strict_convex (Boyd and Vandenberghe sec. 3.1)."""

import pytest

from morie.fn.cvxstx import boyd_strict_convex


def test_cvxstx_basic():
    """For x^2 the midpoint deficit (x - y)^2 / 4 equals (m / 8)(x - y)^2
    exactly with m = 2; |x| is convex, not strictly (chords on one side
    are exact), modulus 0."""
    sq = boyd_strict_convex(lambda x: x[0] ** 2)
    assert sq["convex"] and sq["strictly_convex"] and sq["strongly_convex"]
    assert sq["modulus"] == pytest.approx(2.0, rel=1e-9)
    av = boyd_strict_convex(lambda x: abs(x[0]))
    assert av["convex"] and not av["strictly_convex"]
    assert av["modulus"] == pytest.approx(0.0, abs=1e-12)


def test_cvxstx_edge():
    """-x^2 violates convexity on every chord; a non-callable raises."""
    r = boyd_strict_convex(lambda x: -x[0] ** 2)
    assert not r["convex"] and not r["strictly_convex"]
    assert r["worst_violation"] > 0
    with pytest.raises(TypeError):
        boyd_strict_convex([1.0, 2.0])
