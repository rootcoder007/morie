"""Tests for adamO.adam_optimizer."""

from morie.fn import _array_core as np

from morie.fn.adamO import adam_optimizer


def test_adamO_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    theta = rng.normal(0, 1, 100)
    grad = rng.normal(0, 1, 100)
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    result = adam_optimizer(theta, grad, lr, beta1, beta2, eps)
    # Documented return keys: theta, update, state, m, v (plus step metadata).
    assert "theta" in result
    assert "update" in result
    assert "state" in result
    assert "m" in result
    assert "v" in result
    # Updated parameters must match theta + update.
    new_theta = np.asarray(theta, dtype=float) + np.asarray(result["update"])
    assert np.allclose(result["theta"], new_theta)
    # Independent recomputation of the Adam step (Kingma & Ba, 2015):
    #   m_t = beta1 * m_{t-1} + (1 - beta1) * g
    #   v_t = beta2 * v_{t-1} + (1 - beta2) * g**2
    #   m_hat = m_t / (1 - beta1**t),  v_hat = v_t / (1 - beta2**t)
    #   update = -lr * m_hat / (sqrt(v_hat) + eps)
    g = np.asarray(grad, dtype=float)
    m = (1.0 - beta1) * g
    v = (1.0 - beta2) * g * g
    t = 1
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    expected_update = -lr * m_hat / (np.sqrt(v_hat) + eps)
    assert np.allclose(np.asarray(result["update"]), expected_update)
    assert np.allclose(np.asarray(result["m"]), m)
    assert np.allclose(np.asarray(result["v"]), v)


def test_adamO_edge():
    """Test edge cases."""
    # Single-parameter case using the docstring's two-element example
    # (a valid minimum-shape scenario where theta and grad sizes match).
    rng = np.random.default_rng(42)
    theta = np.array([0.0, 0.0])
    grad = rng.normal(0, 1, 2)
    result = adam_optimizer(theta, grad)
    assert isinstance(result, dict)
    assert "theta" in result
    assert "update" in result
    assert "state" in result
    assert "m" in result
    assert "v" in result
