"""Tests for gdupd.gradient_descent_update."""

from morie.fn import _array_core as np

from morie.fn.gdupd import gradient_descent_update


def test_gdupd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    beta = rng.normal(0, 1, 100)
    grad = rng.normal(0, 1, 100)
    alpha = 0.05
    result = gradient_descent_update(beta, grad, alpha)

    # Independently compute the documented formula: beta - alpha * grad
    expected_beta = np.asarray(beta, dtype=float) - alpha * np.asarray(grad, dtype=float)
    expected_update = -alpha * np.asarray(grad, dtype=float)
    expected_step_norm = float(np.linalg.norm(expected_update))

    assert "beta" in result
    assert "update" in result
    assert "step_norm" in result
    assert np.allclose(result["beta"], expected_beta)
    assert np.allclose(result["update"], expected_update)
    assert result["step_norm"] == expected_step_norm
    assert result["alpha"] == alpha


def test_gdupd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    beta = rng.normal(0, 1, 100)
    grad = rng.normal(0, 1, 100)
    alpha = 0.05
    result = gradient_descent_update(beta, grad, alpha)

    expected_beta = np.asarray(beta, dtype=float) - alpha * np.asarray(grad, dtype=float)
    assert np.allclose(result["beta"], expected_beta)


def test_gdupd_nonpositive_alpha():
    """Negative alpha must raise ValueError per docstring."""
    import pytest
    with pytest.raises(ValueError):
        gradient_descent_update([1.0], [0.5], -0.1)
