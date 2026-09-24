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
    assert math.isfinite(result.payload["estimate"])
    assert math.isfinite(result.payload["objective"])
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
    with pytest.raises(ValueError):
        glpk_lp(c, A, b)
