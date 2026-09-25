"""Tests for smplxs.simplex_lp (primal simplex with Bland's rule)."""

import pytest

from morie.fn.smplxs import simplex_lp


def test_smplxs_basic():
    """max x1 + 2 x2 s.t. x1 + x2 <= 4, x1 - x2 <= 2: optimum x = (0, 4),
    value 8; the dual min 4 y1 + 2 y2 s.t. y1 + y2 >= 1, y1 - y2 >= 2
    ... has y = (2, 0) at the same value (strong duality), and the slack
    of the second constraint is 2 - (0 - 4) = 6."""
    r = simplex_lp([1.0, 2.0], [[1.0, 1.0], [1.0, -1.0]], [4.0, 2.0])
    assert r["status"] == "optimal"
    assert list(r["x"]) == pytest.approx([0.0, 4.0], abs=1e-12)
    assert r["objective"] == pytest.approx(8.0, abs=1e-12)
    assert list(r["dual"]) == pytest.approx([2.0, 0.0], abs=1e-12)
    assert list(r["slack"]) == pytest.approx([0.0, 6.0], abs=1e-12)


def test_smplxs_edge():
    """An objective that can grow without limit is reported unbounded;
    a negative right-hand side is refused."""
    r = simplex_lp([1.0, 1.0], [[1.0, -1.0]], [1.0])
    assert r["status"] == "unbounded"
    with pytest.raises(ValueError):
        simplex_lp([1.0], [[1.0]], [-1.0])
