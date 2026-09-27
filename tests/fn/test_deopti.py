"""Tests for deopti.differential_evolution."""

import math

from morie.fn.deopti import differential_evolution


def _f_simple(x):
    """Simple quadratic objective: f(x) = sum(x_i^2)."""
    return float(sum(v * v for v in x))


def test_deopti_basic():
    """Test basic functionality on a simple quadratic."""
    population = [[0.3, -1.2, 0.8], [1.1, 0.4, -0.6], [-0.9, 0.7, 1.5], [0.2, -0.3, -1.1], [-1.4, 1.0, 0.1]]

    F = 0.8
    CR = 0.9
    generations = 20

    result = differential_evolution(_f_simple, population, F=F, CR=CR, generations=generations)

    # Function returns a RichResult-like object; check its payload keys.
    payload = result.payload

    assert "estimate" in payload
    assert "x" in payload
    assert "population" in payload
    assert "fvals" in payload
    assert "evals" in payload

    # Shape checks on returned objects.
    assert len(payload["population"]) == 5
    assert len(payload["population"][0]) == 3
    assert len(payload["fvals"]) == 5
    assert len(payload["x"]) == 3

    # Evaluate the objective independently on the returned best point.
    x_best = payload["x"]
    expected_f_best = sum(v * v for v in x_best)
    assert math.isclose(payload["estimate"], expected_f_best, rel_tol=1e-9, abs_tol=1e-9)

    # Each returned fval must match the independent evaluation on the
    # corresponding member of the final population.
    for row, fv in zip(payload["population"], payload["fvals"]):
        assert math.isclose(fv, sum(v * v for v in row), rel_tol=1e-9, abs_tol=1e-9)

    # The minimum of fvals equals the reported estimate.
    assert math.isclose(min(payload["fvals"]), payload["estimate"], rel_tol=1e-9, abs_tol=1e-9)

    # Number of evaluations: npop initial + npop*generations subsequent = 5 + 5*20.
    assert payload["evals"] == 5 + 5 * generations


def test_deopti_edge():
    """Test edge cases: tiny population and population of one point."""
    population = [[0.5, -1.1], [-0.8, 0.3], [1.2, 0.9], [-0.4, -1.6]]

    result = differential_evolution(_f_simple, population, F=0.5, CR=0.5, generations=3)
    payload = result.payload

    assert "estimate" in payload
    assert "x" in payload
    assert "population" in payload
    assert "fvals" in payload
    assert "evals" in payload

    assert len(payload["population"]) == 4
    assert len(payload["population"][0]) == 2
    assert len(payload["fvals"]) == 4
    assert len(payload["x"]) == 2

    expected_f_best = sum(v * v for v in payload["x"])
    assert math.isclose(payload["estimate"], expected_f_best, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(min(payload["fvals"]), payload["estimate"], rel_tol=1e-9, abs_tol=1e-9)
    assert payload["evals"] == 4 + 4 * 3
