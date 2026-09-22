"""Tests for advinf.advi."""

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.advinf import advi


def test_advinf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    K = 3
    S = 200
    mu = np.array([0.1, -0.2, 0.3])
    omega = np.array([-0.5, 0.0, 0.5])
    eta = rng.normal(0.0, 1.0, (S, K))

    def logjoint(zeta):
        # Simple analytic log joint: a quadratic in zeta.
        return float(-0.5 * sum(z * z for z in zeta))

    result = advi(mu, omega, eta, logjoint)

    assert isinstance(result, dict)
    # Documented return keys.
    for key in ("elbo", "entropy", "meanlogjoint", "logjoints", "K", "S"):
        assert key in result

    assert result["K"] == K
    assert result["S"] == S
    assert len(result["logjoints"]) == S

    # Independent computation of the ELBO per the docstring formula.
    expected_entropy = sum(omega) + 0.5 * K * (1.0 + math.log(2.0 * math.pi))
    expected_mlj = sum(logjoint([mu[k] + math.exp(omega[k]) * eta[s][k]
                                 for k in range(K)])
                       for s in range(S)) / S
    expected_elbo = expected_mlj + expected_entropy

    assert abs(result["entropy"] - expected_entropy) < 1e-12
    assert abs(result["meanlogjoint"] - expected_mlj) < 1e-12
    assert abs(result["elbo"] - expected_elbo) < 1e-12


def test_advinf_edge():
    """Test edge cases: K=1, and a fully independent recomputation."""
    rng = np.random.default_rng(7)
    K = 1
    S = 50
    mu = np.array([0.0])
    omega = np.array([0.0])
    eta = rng.normal(0.0, 1.0, (S, K))

    calls = []

    def logjoint(zeta):
        calls.append(zeta)
        return float(-0.5 * zeta[0] ** 2)

    result = advi(mu, omega, eta, logjoint)

    assert isinstance(result, dict)
    assert result["K"] == K
    assert result["S"] == S
    assert len(calls) == S  # logjoint evaluated once per draw.

    expected_entropy = sum(omega) + 0.5 * K * (1.0 + math.log(2.0 * math.pi))
    expected_mlj = sum(logjoint([mu[k] + math.exp(omega[k]) * eta[s][k]
                                 for k in range(K)])
                       for s in range(S)) / S
    expected_elbo = expected_mlj + expected_entropy

    assert abs(result["entropy"] - expected_entropy) < 1e-12
    assert abs(result["meanlogjoint"] - expected_mlj) < 1e-12
    assert abs(result["elbo"] - expected_elbo) < 1e-12
