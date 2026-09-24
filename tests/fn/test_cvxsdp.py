"""Tests for cvxsdp.boyd_sdp."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.cvxsdp import boyd_sdp


def test_cvxsdp_basic():
    """Test basic functionality: min largest eigenvalue of a 2x2 matrix via LMI."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    c = [1.0]
    F = [-A, np.eye(2)]
    result = boyd_sdp(c, F)
    for key in ("x", "objective", "slack", "eigenvalues", "gap_bound",
                "feasible", "strictly_feasible", "active", "converged"):
        assert key in result
    assert math.isfinite(float(result["objective"]))
    assert len(result["x"]) == 1


def test_cvxsdp_edge():
    """Test edge case: larger 3x3 symmetric matrix with a relaxed tolerance."""
    A = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.0]])
    c = [1.0]
    F = [-A, np.eye(3)]
    result = boyd_sdp(c, F, tol=1e-6)
    for key in ("x", "objective", "slack", "eigenvalues", "gap_bound",
                "feasible", "strictly_feasible", "active", "converged"):
        assert key in result
    assert math.isfinite(float(result["objective"]))
    assert len(result["x"]) == 1
