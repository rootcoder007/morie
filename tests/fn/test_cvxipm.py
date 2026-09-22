"""Tests for cvxipm.boyd_interior_point."""

from morie.fn import _array_core as np

from morie.fn.cvxipm import boyd_interior_point


def test_cvxipm_basic():
    """Test basic functionality: minimise x^2/2 subject to x >= 1."""
    # Objective: f0(x) = x[0]^2 / 2
    obj = lambda x: 0.5 * x[0] ** 2
    # Inequality constraint: 1 - x[0] <= 0  (i.e., x >= 1)
    con = [lambda x: 1.0 - x[0]]
    # Strictly feasible start: x0 = 2.0 satisfies 1 - 2 = -1 < 0
    x0 = [2.0]

    result = boyd_interior_point(obj, con, x0)

    assert isinstance(result, dict)
    # The documented RichResult keys
    assert "x" in result
    assert "objective" in result
    assert "gap_bound" in result
    assert "path" in result
    assert "t_values" in result
    assert "constraints" in result
    assert "strictly_feasible" in result
    assert "newton_steps" in result
    assert "outer" in result
    assert "converged" in result

    # The optimum is at the boundary x = 1, value = 1/2
    assert abs(float(result["x"][0]) - 1.0) < 1e-4
    assert abs(float(result["objective"]) - 0.5) < 1e-4

    # gap_bound = m / t for the final centering t, which is < tol = 1e-6
    assert float(result["gap_bound"]) < 1e-5

    # converged flag must be True
    assert bool(result["converged"]) is True

    # All iterates along the path are STRICTLY feasible (> 1.0)
    assert bool(np.all(result["path"] > 1.0))

    # Final constraint value is <= 0 (feasible)
    assert float(result["constraints"][0]) <= 0.0

    # strictly_feasible should be True
    assert bool(result["strictly_feasible"]) is True

    # Expected objective at the optimal x (independent recomputation)
    x_star = float(result["x"][0])
    expected_obj = 0.5 * x_star ** 2
    assert abs(float(result["objective"]) - expected_obj) < 1e-12


def test_cvxipm_edge():
    """Test edge cases: single outer iteration gives the golden-ratio centered point."""
    # Same problem as test_cvxipm_basic but capped at max_outer=1
    obj = lambda x: 0.5 * x[0] ** 2
    con = [lambda x: 1.0 - x[0]]
    x0 = [2.0]

    result = boyd_interior_point(obj, con, x0, max_outer=1)

    assert isinstance(result, dict)
    assert "x" in result
    assert "gap_bound" in result
    assert "objective" in result
    assert "converged" in result

    # After one outer iteration with t = 1, the centered point solves
    # x^2 - x - 1 = 0, so x is the golden ratio (1 + sqrt(5)) / 2
    expected_golden = (1.0 + 5.0 ** 0.5) / 2.0
    assert abs(float(result["x"][0]) - expected_golden) < 1e-5

    # Independent recomputation of objective
    x_c = float(result["x"][0])
    expected_obj = 0.5 * x_c ** 2
    assert abs(float(result["objective"]) - expected_obj) < 1e-12

    # gap_bound = m / t = 1 / 1 = 1.0 for the final centered t
    assert abs(float(result["gap_bound"]) - 1.0) < 1e-12

    # The true suboptimality (objective - p*) is bounded by gap_bound
    assert bool(float(result["objective"]) - 0.5 <= float(result["gap_bound"]))

    # Iterates remain strictly feasible
    assert bool(np.all(result["path"] > 1.0))

    # outer count is exactly 1
    assert int(result["outer"]) == 1

    # converged must be False (gap_bound > tol)
    assert bool(result["converged"]) is False
