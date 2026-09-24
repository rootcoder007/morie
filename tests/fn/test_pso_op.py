"""Tests for pso_op.particle_swarm."""

import math

from morie.fn.pso_op import particle_swarm


def test_pso_op_basic():
    """Test basic functionality."""
    def sphere(x):
        return sum(xi * xi for xi in x)

    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

    result = particle_swarm(sphere, bounds, n_particles=10, maxiter=50)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value" in result
    assert "x" in result
    assert "n_eval" in result
    assert "n_particles" in result
    assert "maxiter" in result
    assert "d" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0
    assert result["n_particles"] == 10
    assert result["maxiter"] == 50
    assert result["d"] == 3
    assert len(result["x"]) == 3


def test_pso_op_edge():
    """Test edge cases."""
    def quadratic(x):
        return x[0] * x[0]

    bounds = [(-10.0, 10.0)]
    result = particle_swarm(quadratic, bounds, n_particles=3, maxiter=5)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0
    assert result["d"] == 1
    assert len(result["x"]) == 1
    assert result["n_particles"] == 3
    assert result["maxiter"] == 5
