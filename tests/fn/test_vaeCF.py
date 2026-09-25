"""Tests for vaeCF.vae_cf (Mult-VAE, Liang et al. 2018)."""

import math

import pytest

from morie.fn.vaeCF import vae_cf


R = [[1, 0, 2, 0, 1, 0], [0, 1, 0, 0, 3, 1], [2, 2, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0]]


def test_vaeCF_basic():
    """With w_scale = 0 the decoder logits are constant, so pi is uniform
    and log p(x_u | z) = sum_i x_ui log(1/I) exactly for every draw.  The
    KL term is the Gaussian closed form 1/2 sum(mu^2 + exp(logvar) - 1 -
    logvar) at the encoder's own mu, logvar; ELBO = loglik - beta KL."""
    r = vae_cf(R, K=2, latent_dim=2, beta=0.3, w_scale=0.0)
    I = 6
    for u, row in enumerate(R):
        assert r["loglik_per_user"][u] == pytest.approx(sum(row) * math.log(1 / I), abs=1e-12)
        mu, lv = r["mu"][u], r["logvar"][u]
        kl = 0.5 * sum(m * m + math.exp(v) - 1 - v for m, v in zip(mu, lv))
        assert r["kl_per_user"][u] == pytest.approx(kl, abs=1e-12)
        assert r["elbo_per_user"][u] == pytest.approx(r["loglik_per_user"][u] - 0.3 * kl, abs=1e-12)


def test_vaeCF_edge():
    """Recall@K for one user: hits among the top-K of the returned
    ranking over min(K, #relevant); negative clicks, K out of range and
    a mis-shaped relevance raise."""
    r = vae_cf(R, K=3, w_scale=1.0)
    for u, row in enumerate(R):
        top = [int(i) - 1 for i in r["ranking"][u][:3]]
        rel = [1 if v > 0 else 0 for v in row]
        assert r["recall_per_user"][u] == pytest.approx(sum(rel[i] for i in top) / min(3, sum(rel)), abs=1e-15)
    with pytest.raises(ValueError):
        vae_cf([[1, -1], [0, 1]])
    with pytest.raises(ValueError):
        vae_cf(R, K=7)
    with pytest.raises(ValueError):
        vae_cf(R, relevance=[[1, 0]])
