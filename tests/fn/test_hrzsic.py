"""Tests for hrzsic.horowitz_sim_identification."""

from morie.fn import _array_core as np

from morie.fn.hrzsic import horowitz_sim_identification


def test_hrzsic_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sim_identification(x, beta)
    assert isinstance(result, dict)
    assert "identified" in result


def test_hrzsic_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_sim_identification(x, beta)
    assert isinstance(result, dict)
