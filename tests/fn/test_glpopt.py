"""Tests for glpopt.glpk_lp."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.glpopt import glpk_lp


def test_glpopt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 4, 3
    c = list(rng.normal(0, 1, n))
    A = [list(rng.normal(0, 1, n)) for _ in range(m)]
    b = [abs(float(rng.normal(0, 1))) + 0.5 for _ in range(m)]
    result = glpk_lp(c, A, b)
    assert isinstance(result.payload, dict)
    assert "estimate" in result.payload
    assert "x" in result.payload
    assert "objective" in result.payload
    assert "dual" in result.payload
    assert "status" in result.payload
    assert "n" in result.payload
    assert "iterations" in result.payload
    # an objective exists only at an optimum; an unbounded problem has none
    if result.payload["status"] == "optimal":
        assert math.isfinite(result.payload["estimate"])
        assert math.isfinite(result.payload["objective"])
    else:
        assert math.isnan(result.payload["objective"])
    assert result.payload["n"] == n
    assert len(result.payload["x"]) == n
    assert len(result.payload["dual"]) == m
    assert isinstance(result.payload["iterations"], int)
    assert result.payload["status"] in ("optimal", "unbounded")


def test_glpopt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, m = 3, 2
    c = list(rng.normal(0, 1, n))
    A = [list(rng.normal(0, 1, n)) for _ in range(m)]
    b = [-1.0, 1.0]
    # a negative right-hand side goes through phase one
    r = glpk_lp(c, A, b)
    assert r.payload["status"] in ("optimal", "unbounded", "infeasible")
    if r.payload["status"] == "optimal":
        x = r.payload["x"]
        for i in range(m):
            assert sum(A[i][j] * x[j] for j in range(n)) <= b[i] + 1e-9
        assert abs(r.payload["dual_objective"] - r.payload["objective"]) < 1e-9
    with pytest.raises(ValueError):
        glpk_lp(c, A, [1.0])


def test_glpopt_phase_one():
    """x1 + x2 >= 2, x1 <= 5: min x1 + 3 x2 is 2 at (2, 0); the same as the R arm."""
    r = glpk_lp([1, 3], [[-1, -1], [1, 0]], [-2, 5]).payload
    assert r["status"] == "optimal"
    assert r["x"] == pytest.approx([2.0, 0.0], abs=1e-12)
    assert r["objective"] == pytest.approx(2.0, abs=1e-12)
    assert r["dual_objective"] == pytest.approx(r["objective"], abs=1e-12)
    assert all(v <= 1e-12 for v in r["dual"])
    # x1 >= 3 and x1 <= 2 cannot both hold
    assert glpk_lp([1, 1], [[-1, 0], [1, 0]], [-3, 2]).payload["status"] == "infeasible"
    # hitting the cap is reported, not passed off as an optimum
    wy = glpk_lp([-3, -5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18], max_iter=1)
    assert wy.payload["status"] == "iteration_limit"
