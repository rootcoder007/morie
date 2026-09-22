"""Tests for betvae.beta_vae_disentangle."""

from morie.fn import _array_core as np

from morie.fn.betvae import beta_vae_disentangle


def test_betvae_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    d = 100
    J = 8
    x = rng.normal(0, 1, d)
    xhat = rng.normal(0, 1, d)
    mu = rng.normal(0, 1, J)
    logvar = rng.normal(0, 1, J)
    beta = 0.8
    result = beta_vae_disentangle(x, xhat, mu, logvar, beta)
    assert isinstance(result, dict)
    # Documented keys from the RichResult payload.
    for key in ("objective", "recon", "kl", "klper", "penalty",
                "beta", "J", "d"):
        assert key in result, f"missing key: {key}"

    # Recompute the expected quantities from the documented formulas
    # using plain arithmetic on the same inputs.
    nv = 1.0
    rec_expected = -sum((x[i] - xhat[i]) ** 2 for i in range(d)) / (2.0 * nv) \
        - 0.5 * d * np.log(2.0 * np.pi * nv)
    per_expected = [0.5 * (mu[j] * mu[j] + np.exp(logvar[j]) - 1.0 - logvar[j])
                    for j in range(J)]
    kl_expected = sum(per_expected)
    pen_expected = beta * kl_expected
    obj_expected = rec_expected - pen_expected

    assert result["d"] == d
    assert result["J"] == J
    assert result["beta"] == beta
    assert np.allclose(result["recon"], rec_expected)
    assert np.allclose(result["kl"], kl_expected)
    assert np.allclose(result["klper"], per_expected)
    assert np.allclose(result["penalty"], pen_expected)
    assert np.allclose(result["objective"], obj_expected)


def test_betvae_edge():
    """Test edge cases (capacity variant and no capacity)."""
    rng = np.random.default_rng(42)
    d = 50
    J = 5
    x = rng.normal(0, 1, d)
    xhat = rng.normal(0, 1, d)
    mu = rng.normal(0, 1, J)
    logvar = rng.normal(0, 1, J)

    # Plain beta penalty when capacity is None.
    result = beta_vae_disentangle(x, xhat, mu, logvar, beta=2.0)
    assert isinstance(result, dict)
    assert np.allclose(result["penalty"], 2.0 * result["kl"])
    assert np.allclose(result["objective"], result["recon"] - result["penalty"])

    # Capacity variant with explicit gamma.
    C = 10.0
    gamma = 1.5
    result_cap = beta_vae_disentangle(x, xhat, mu, logvar, beta=4.0,
                                      capacity=C, gamma=gamma)
    assert np.allclose(result_cap["penalty"], gamma * abs(result_cap["kl"] - C))
    assert np.allclose(result_cap["objective"],
                       result_cap["recon"] - result_cap["penalty"])
