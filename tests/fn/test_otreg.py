"""Tests for otreg.ot_regularised_dual."""

from morie.fn import _array_core as np

from morie.fn.otreg import ot_regularised_dual


def test_otreg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, m = 40, 50
    a = list(rng.uniform(0, 1, n))
    b = list(rng.uniform(0, 1, m))
    target = 1.0
    a_sum = sum(a)
    b_sum = sum(b)
    a = [x * target / a_sum for x in a]
    b = [x * target / b_sum for x in b]
    C = rng.uniform(0, 1, (n, m))
    epsilon = 0.1
    max_iter = 100
    result = ot_regularised_dual(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
    assert "f" in result
    assert "g" in result
    assert "dual_value" in result
    assert "primal_cost" in result
    assert result["n"] == n
    assert result["m"] == m
    assert len(result["f"]) == n
    assert len(result["g"]) == m


def test_otreg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, m = 5, 5
    a = [0.2] * n
    b = [0.2] * m
    C = rng.uniform(0, 1, (n, m))
    epsilon = 0.01
    max_iter = 50
    result = ot_regularised_dual(a, b, C, epsilon, max_iter)
    assert isinstance(result, dict)
    assert "f" in result
    assert "g" in result
    assert "dual_value" in result
    assert "primal_cost" in result
    assert result["n"] == n
    assert result["m"] == m
