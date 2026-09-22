"""Tests for btparm.boot_parametric."""

from morie.fn import _array_core as np

from morie.fn.btparm import boot_parametric


def test_btparm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    theta_hat = np.array([0.0, 1.0])
    rvs_fn = lambda th, n, g: rng.normal(th[0], th[1], n)
    stat = lambda x: np.mean(x)
    B = 200
    n = 100
    result = boot_parametric(theta_hat, rvs_fn, stat, B, n)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "lo" in result
    assert "hi" in result
    assert "var_closed" in result
    assert "n" in result
    assert "B" in result
    assert "theta_b" in result
    assert result["n"] == n
    assert result["B"] == B


def test_btparm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    theta_hat = np.array([0.0, 1.0])
    rvs_fn = lambda th, n, g: rng.normal(th[0], th[1], n)
    stat = lambda x: np.mean(x)
    B = 200
    n = 100
    result = boot_parametric(theta_hat, rvs_fn, stat, B, n)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "lo" in result
    assert "hi" in result
    assert "var_closed" in result
    assert "n" in result
    assert "B" in result
    assert "theta_b" in result
    assert result["n"] == n
    assert result["B"] == B
