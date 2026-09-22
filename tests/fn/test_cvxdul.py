"""Tests for cvxdul.boyd_lagrangian."""

from morie.fn import _array_core as np

from morie.fn.cvxdul import boyd_lagrangian


def test_cvxdul_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    f0 = float(rng.normal())
    f = rng.normal(0, 1, 100)
    h = rng.normal(0, 1, 100)
    lambda_ = rng.uniform(0, 1, 100)
    nu = rng.normal(0, 1, 100)
    result = boyd_lagrangian(f0, f, h, lambda_, nu)
    assert isinstance(result, dict)
    assert "value" in result
    assert "objective" in result
    assert "ineq_term" in result
    assert "eq_term" in result
    assert "feasible" in result
    assert "complementary_slackness" in result
    expected = f0 + float(lambda_ @ f) + float(nu @ h)
    assert result["value"] == expected
    assert result["objective"] == f0
    assert result["ineq_term"] == float(lambda_ @ f)
    assert result["eq_term"] == float(nu @ h)
    assert result["feasible"] is False


def test_cvxdul_edge():
    """Test edge cases."""
    result = boyd_lagrangian(3.0, f=[-1.0, -2.0], h=[0.0])
    assert isinstance(result, dict)
    assert result["value"] == 3.0
    assert result["feasible"] is True

    result2 = boyd_lagrangian(3.0, f=[-1.0, -2.0], lambda_=[2.0, 1.0])
    assert isinstance(result2, dict)
    assert result2["value"] == -1.0
    assert result2["complementary_slackness"] > 0
