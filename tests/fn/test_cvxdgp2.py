"""Tests for cvxdgp2.boyd_duality_gap."""

import math

from morie.fn import _array_core as np

from morie.fn.cvxdgp2 import boyd_duality_gap


def test_cvxdgp2_basic():
    """Test basic functionality: scalar primal and dual produce documented keys."""
    primal = 5.0
    dual = 4.0
    result = boyd_duality_gap(primal, dual)

    expected_gap = primal - dual
    expected_relative = expected_gap / max(abs(primal), 1e-300)

    assert isinstance(result, dict)
    assert result["gap"] == expected_gap
    assert result["relative_gap"] == expected_relative
    assert result["closed"] is (expected_gap <= 1e-09)
    assert result["strong_duality"] is (expected_gap <= 1e-09)
    assert result["bracket"] == (dual, primal)
    assert result["primal"] == primal
    assert result["dual"] == dual
    assert result["method"] == "boyd_duality_gap"


def test_cvxdgp2_edge():
    """Test edge cases: closed gap implies strong duality; zero gap."""
    primal = 2.0
    dual = 2.0
    result = boyd_duality_gap(primal, dual)

    expected_gap = primal - dual

    assert isinstance(result, dict)
    assert result["gap"] == expected_gap
    assert result["strong_duality"] is True
    assert result["closed"] is True
    assert result["bracket"] == (dual, primal)
    assert math.isclose(result["relative_gap"], 0.0, abs_tol=1e-300)
