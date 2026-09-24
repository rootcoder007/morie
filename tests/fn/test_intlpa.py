"""Tests for intlpa.interior_point_lp."""

from morie.fn import _array_core as np

from morie.fn.intlpa import interior_point_lp

import math

import pytest


def test_intlpa_basic():
    """Test basic functionality with a feasible LP."""
    n = 3
    m = 2
    c = [1.0, 1.0, 1.0]
    # Use orthogonal rows to guarantee a well-conditioned Hessian
    A = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    x0 = [2.0, 2.0, 2.0]
    # Strictly feasible: b - A @ x0 > 0
    b = [3.0, 3.0]
    tau = 0.5
    result = interior_point_lp(c, A, b, x0, tau)
    assert hasattr(result, "estimate")
    assert math.isfinite(result.estimate)
    assert hasattr(result, "x")
    assert len(result.x) == n
    assert hasattr(result, "objective")
    assert math.isfinite(result.objective)


def test_intlpa_edge():
    """Test that tau=0 raises ValueError."""
    n = 3
    m = 2
    c = [1.0, 1.0, 1.0]
    A = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    x0 = [2.0, 2.0, 2.0]
    b = [3.0, 3.0]
    with pytest.raises(ValueError):
        interior_point_lp(c, A, b, x0, tau=0)
