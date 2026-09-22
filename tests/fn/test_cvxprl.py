"""Tests for cvxprl.boyd_perspective."""

from morie.fn import _array_core as np

from morie.fn.cvxprl import boyd_perspective


def test_cvxprl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    f = lambda z: z ** 2
    x = rng.normal(0, 1, 100)
    t = np.linspace(0.5, 10, 100)
    result = boyd_perspective(f, x, t)
    assert isinstance(result, dict)
    assert "value" in result
    assert "ratio" in result
    assert "f_ratio" in result
    assert "homogeneous" in result
    expected_value = t * f(x / t)
    assert np.allclose(result["value"], expected_value)
    assert np.allclose(result["ratio"], x / t)
    assert np.allclose(result["f_ratio"], f(x / t))
    assert result["homogeneous"] is True


def test_cvxprl_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    f = lambda z: z ** 2
    x = rng.normal(0, 1, 100)
    t = np.linspace(0.5, 10, 100)
    result = boyd_perspective(f, x, t)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["homogeneous"] is True
