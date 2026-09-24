"""Tests for penmth.penalty_method."""

import math
import pytest

from morie.fn import _array_core as np
from morie.fn.penmth import penalty_method


def test_penmth_basic():
    """Test basic functionality."""
    # Define a simple objective: f(x) = sum(x_i^2)
    def f(x):
        return sum(v * v for v in x)

    # Define a single inequality constraint: sum(x) - 1 <= 0  => sum(x) <= 1
    def g1(x):
        return sum(x) - 1.0

    constraints = [g1]

    # Starting point: a small list of floats (e.g., 3 variables)
    x0 = [0.5, 0.5, 0.5]

    mu = 1.0  # positive penalty weight

    # Use small iteration counts to keep the test fast
    result = penalty_method(f, constraints, x0, mu, n_outer=2, growth=2.0, n_inner=50)

    # The result should be a dict-like object (RichResult)
    assert isinstance(result, dict)

    # Check that all expected keys are present
    expected_keys = {
        "x",
        "f",
        "penalty",
        "violation",
        "max_violation",
        "q",
        "mu",
        "n_outer",
        "n_inner",
        "method",
    }
    assert expected_keys.issubset(set(result.keys()))

    # Check that x has same length as starting point
    assert len(result["x"]) == len(x0)

    # f, penalty, q, mu should be finite numbers
    assert math.isfinite(result["f"])
    assert math.isfinite(result["penalty"])
    assert math.isfinite(result["q"])
    assert math.isfinite(result["mu"])

    # violation should be a list of floats (or empty)
    assert isinstance(result["violation"], list)
    for vi in result["violation"]:
        assert math.isfinite(vi)

    # max_violation should be finite
    assert math.isfinite(result["max_violation"])

    # n_outer and n_inner should be integers
    assert isinstance(result["n_outer"], int)
    assert isinstance(result["n_inner"], int)

    # method should be a string
    assert isinstance(result["method"], str)


def test_penmth_edge():
    """Test edge cases: invalid mu and growth raise ValueError."""
    def f(x):
        return sum(v * v for v in x)

    constraints = []

    x0 = [0.0, 0.0]

    # mu must be positive
    with pytest.raises(ValueError):
        penalty_method(f, constraints, x0, 0.0)

    with pytest.raises(ValueError):
        penalty_method(f, constraints, x0, -1.0)

    # growth must exceed 1
    with pytest.raises(ValueError):
        penalty_method(f, constraints, x0, 1.0, growth=0.5)

    with pytest.raises(ValueError):
        penalty_method(f, constraints, x0, 1.0, growth=1.0)
