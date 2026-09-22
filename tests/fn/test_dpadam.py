"""Tests for dpadam.dp_adam."""

from morie.fn import _array_core as np

from morie.fn.dpadam import dp_adam


def test_dpadam_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # grads must be 2-D (B, p): per-example gradients
    grads = rng.normal(0, 1, (16, 4))
    C = 1.0
    sigma = 1.0
    lr = 0.001
    betas = (0.9, 0.999)
    result = dp_adam(grads, C=C, sigma=sigma, lr=lr, betas=betas, seed=42)
    assert isinstance(result, dict)
    # Documented keys returned by dp_adam
    for key in ("update", "state", "private_gradient",
                "signal_to_noise", "clipped_fraction"):
        assert key in result
    # Independent arithmetic check of the first Adam step on the
    # privatised gradient (sigma=0, so private_gradient == clipped mean).
    expected = -lr * (grads.mean(axis=0)) / (np.sqrt(grads.mean(axis=0) ** 2) + 1e-8)
    assert bool(np.allclose(np.asarray(result["update"]).reshape(-1),
                            expected.reshape(-1), atol=1e-10))


def test_dpadam_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    grads = rng.normal(0, 1, (16, 4))
    # Second call threads state from the first; t should increment to 2.
    r1 = dp_adam(grads, C=1.0, sigma=0.0, lr=0.01, seed=0)
    r2 = dp_adam(grads, C=1.0, sigma=0.0, lr=0.01, state=r1["state"], seed=0)
    assert isinstance(r2, dict)
    assert int(r2["state"]["t"]) == 2

    # signal_to_noise must be a finite scalar
    snr = r1["signal_to_noise"]
    assert isinstance(snr, float)
    assert np.isfinite(snr)

    # Higher noise multiplier => lower signal-to-noise (with same seed)
    lo = dp_adam(grads, C=1.0, sigma=0.1, seed=1)["signal_to_noise"]
    hi = dp_adam(grads, C=1.0, sigma=10.0, seed=1)["signal_to_noise"]
    assert bool(hi < lo)
