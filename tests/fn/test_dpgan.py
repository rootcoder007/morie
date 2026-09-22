"""Tests for dpgan.dp_gan."""

from morie.fn import _array_core as np

from morie.fn.dpgan import dp_gan


def test_dpgan_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    G = rng.normal(size=(64, 6)) * 0.1
    result = dp_gan(G, C=1.0, sigma=1.0, lr=0.05, n_disc_steps=5, seed=0)

    # Output is a RichResult (dict-like) with the documented payload keys.
    assert isinstance(result, dict)
    assert result["generator_is_free"] is True
    assert int(result["steps_to_account"]) == 5
    assert "private_gradient" in result
    assert "disc_update" in result
    assert "clipped_fraction" in result

    # The privatised gradient must equal DP-SGD on the same input, since
    # dp_gan is exactly DP-SGD with an extra -lr*g step accounted for.
    from morie.fn.dpsgd import dp_sgd
    ref = dp_sgd(G, C=1.0, sigma=1.0, lr=1.0, seed=0)["private_gradient"]
    assert bool(np.allclose(result["private_gradient"], ref))

    # disc_update = -lr * private_gradient, by independent arithmetic.
    g = np.asarray(result["private_gradient"], dtype=float)
    assert bool(np.allclose(result["disc_update"], -0.05 * g))


def test_dpgan_edge():
    """Test edge case: minimal batch, fully-clipping inputs."""
    rng = np.random.default_rng(7)
    # Per-example gradients of shape (B, p); here B=1 to exercise the batch
    # floor and clipping paths.
    G = rng.normal(size=(1, 4))
    result = dp_gan(G, C=0.5, sigma=0.0, lr=0.1, n_disc_steps=1, seed=123)

    assert isinstance(result, dict)
    assert result["generator_is_free"] is True
    assert int(result["steps_to_account"]) == 1

    # With sigma=0, clipped gradients => private_gradient and disc_update
    # are exactly the (sign-flipped, scaled) per-example gradient.
    g = np.asarray(result["private_gradient"], dtype=float)
    assert bool(np.allclose(result["disc_update"], -0.1 * g))
