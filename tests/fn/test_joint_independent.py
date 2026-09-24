"""Tests for joint_independent.joint_independent."""

import math

from morie.fn import _array_core as np

from morie.fn.joint_independent import (
    joint_independent,
)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e9_basic():
    """Test basic functionality with an independent joint pmf."""
    rng = np.random.default_rng(42)
    n_x = 3
    n_y = 4
    # Generate random marginals via integer counts
    counts_x = rng.integers(0, 10, n_x)
    marg_x = [float(c) for c in counts_x]
    total_x = sum(marg_x)
    marg_x = [v / total_x for v in marg_x]
    counts_y = rng.integers(0, 10, n_y)
    marg_y = [float(c) for c in counts_y]
    total_y = sum(marg_y)
    marg_y = [v / total_y for v in marg_y]
    # Construct joint pmf as outer product (independent)
    joint = [[marg_x[i] * marg_y[j] for j in range(n_y)] for i in range(n_x)]
    result = joint_independent(joint)
    assert isinstance(result, dict)
    assert "independent" in result
    assert result["independent"] is True
    assert "marginal_x" in result
    assert "marginal_y" in result
    # Marginals should sum to 1
    assert math.isclose(sum(result["marginal_x"]), 1.0, abs_tol=1e-9)
    assert math.isclose(sum(result["marginal_y"]), 1.0, abs_tol=1e-9)


def test_david_j_morin_probability_for_the_enthusiastic_beginner3e9_edge():
    """Test edge case with a dependent joint pmf."""
    # A 2x2 joint pmf that does not factorize as product of marginals
    joint = [[0.5, 0.0],
             [0.0, 0.5]]
    result = joint_independent(joint)
    assert isinstance(result, dict)
    assert "independent" in result
    assert result["independent"] is False
    assert "marginal_x" in result
    assert "marginal_y" in result
    assert math.isclose(sum(result["marginal_x"]), 1.0, abs_tol=1e-9)
    assert math.isclose(sum(result["marginal_y"]), 1.0, abs_tol=1e-9)
