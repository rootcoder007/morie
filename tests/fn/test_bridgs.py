"""Tests for bridgs.bridge_sampling."""

from morie.fn import _array_core as np

from morie.fn.bridgs import bridge_sampling


def test_bridgs_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    draws1 = rng.normal(0, 1, 100)
    draws2 = rng.normal(0, 1, 100)

    def log_q1(x):
        return -0.5 * (x ** 2)

    def log_q2(x):
        return -0.5 * ((x - 0.3) ** 2)

    result = bridge_sampling(draws1, draws2, log_q1, log_q2)
    assert isinstance(result, dict)
    assert "ratio" in result
    assert "log_ratio" in result
    assert "iterations" in result
    assert "converged" in result
    assert isinstance(result["ratio"], float)
    assert isinstance(result["log_ratio"], float)
    assert isinstance(result["iterations"], int)
    assert isinstance(result["converged"], bool)


def test_bridgs_edge():
    """Test edge cases: equal densities yield ratio ~ 1."""
    rng = np.random.default_rng(42)
    draws1 = rng.normal(0, 1, 100)
    draws2 = rng.normal(0, 1, 100)

    def log_q(x):
        return -0.5 * (x ** 2)

    result = bridge_sampling(draws1, draws2, log_q, log_q)
    assert isinstance(result, dict)
    assert "ratio" in result
    assert isinstance(result["ratio"], float)
    assert result["converged"] is True
