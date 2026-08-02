"""Tests for vaenc.vae_elbo."""

from morie.fn import _array_core as np
import pytest

from morie.fn.vaenc import vae_elbo


def test_vaenc_kl_vanishes_at_the_standard_normal():
    """KL(q||N(0,1)) = 0 exactly when mu = 0 and log_var = 0. Anything else
    means the closed form is wrong."""
    x = np.zeros(8)
    r = vae_elbo(x, x, np.zeros(8), np.zeros(8))
    assert float(r["kl_divergence"]) == pytest.approx(0.0, abs=1e-12)


def test_vaenc_kl_matches_the_closed_form():
    """-0.5 * sum(1 + log_var - mu^2 - exp(log_var)), Kingma & Welling (2014)."""
    rng = np.random.default_rng(0)
    mu = rng.normal(size=6)
    log_var = rng.normal(size=6) * 0.5
    x = np.zeros(6)
    r = vae_elbo(x, x, mu, log_var, reduction="sum")
    want = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    assert float(r["kl_divergence"]) == pytest.approx(want, rel=1e-9)


def test_vaenc_elbo_is_minus_recon_minus_kl():
    rng = np.random.default_rng(1)
    x = rng.normal(size=10)
    r = vae_elbo(x, x + 0.1, rng.normal(size=10), np.zeros(10))
    assert float(r["elbo"]) == pytest.approx(-(r["recon_loss"] + r["kl_divergence"]), rel=1e-9)
    assert float(r["loss"]) == pytest.approx(-float(r["elbo"]), rel=1e-12)


def test_vaenc_perfect_reconstruction_has_zero_recon_loss():
    rng = np.random.default_rng(2)
    x = rng.normal(size=12)
    r = vae_elbo(x, x, np.zeros(12), np.zeros(12))
    assert float(r["recon_loss"]) == pytest.approx(0.0, abs=1e-12)
    assert float(r["elbo"]) == pytest.approx(0.0, abs=1e-12)


def test_vaenc_rejects_an_unknown_reduction():
    x = np.zeros(4)
    with pytest.raises(ValueError, match="reduction"):
        vae_elbo(x, x, x, x, reduction="median")
