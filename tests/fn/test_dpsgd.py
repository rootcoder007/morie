"""Tests for dpsgd.dp_sgd."""

from morie.fn import _array_core as np

from morie.fn.dpsgd import dp_sgd


def test_dpsgd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    grads = rng.normal(0, 1, (100, 5))
    C = 1.0
    sigma = 1.0
    lr = 0.1
    result = dp_sgd(grads, C, sigma, lr, seed=0)
    assert isinstance(result, dict)
    assert "private_gradient" in result
    assert "update" in result
    assert "clipped_fraction" in result
    assert "noise_sd" in result

    # Recompute the expected private gradient independently from the formula
    # and compare it to the value reported by the function.
    G = np.asarray(grads, dtype=float)
    norms = np.linalg.norm(G, axis=1)
    factor = np.minimum(1.0, C / np.maximum(norms, 1e-12))
    Gc = G * factor[:, None]
    rng2 = np.random.default_rng(0)
    noise = rng2.normal(0.0, sigma * C, G.shape[1]) if sigma > 0 else np.zeros(G.shape[1])
    expected_private = (Gc.sum(axis=0) + noise) / G.shape[0]
    expected_update = -lr * expected_private
    expected_clipped = float(np.mean(norms > C))
    expected_noise_sd = float(sigma * C)

    assert np.allclose(result["private_gradient"], expected_private)
    assert np.allclose(result["update"], expected_update)
    assert float(result["clipped_fraction"]) == expected_clipped
    assert float(result["noise_sd"]) == expected_noise_sd


def test_dpsgd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    grads = rng.normal(0, 1, (100, 5))
    C = 1.0
    sigma = 1.0
    lr = 0.1
    result = dp_sgd(grads, C, sigma, lr, seed=0)
    assert isinstance(result, dict)
    assert "private_gradient" in result
    assert result["clipped_fraction"] >= 0.0
    assert result["clipped_fraction"] <= 1.0
    assert float(result["noise_sd"]) == float(sigma * C)
