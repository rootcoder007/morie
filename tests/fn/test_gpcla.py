"""Tests for gpcla.gp_classification."""

import math

from morie.fn import _array_core as np

from morie.fn.gpcla import gp_classification


def test_gpcla_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5)).tolist()
    y01 = rng.normal(0, 1, 100)
    y = [1 if v > 0 else 0 for v in y01]
    rng_t = np.random.default_rng(43)
    X_test = rng_t.normal(0, 1, 30).tolist()
    kernel = lambda u: math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    result = gp_classification(X, y, X_test, kernel)
    # Independent check on the documented probit prediction
    # formula p = Phi(mu / sqrt(1 + var)) using the returned
    # latent_mean and latent_var.
    assert isinstance(result.title, str)
    assert result.title == "GP classification (Laplace)"
    # expected probability computed without calling the function's
    # helpers, just the documented probit transform on its outputs
    sd_latent = result.payload["latent_var"]
    mu_latent = result.payload["latent_mean"]
    p = result.payload["p"]
    # Reproduce the probit transform: standard normal CDF
    def _phi(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    expected = [_phi(mu_latent[j] / math.sqrt(1.0 + sd_latent[j]))
                for j in range(len(X_test))]
    for a, b in zip(expected, p):
        assert abs(a - b) < 1e-9
    assert result.payload["estimate"] == p[0]
    assert all(v in (0, 1) for v in result.payload["predicted"])
    assert result.payload["n"] == 100
    assert "objective" in result.payload


def test_gpcla_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5)).tolist()
    y01 = rng.normal(0, 1, 100)
    y = [1 if v > 0 else 0 for v in y01]
    rng_t = np.random.default_rng(43)
    X_test = rng_t.normal(0, 1, 30).tolist()
    kernel = lambda u: math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    result = gp_classification(X, y, X_test, kernel)
    assert isinstance(result.title, str)
    assert len(result.payload["p"]) == len(X_test)
