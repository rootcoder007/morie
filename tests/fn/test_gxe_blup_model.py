"""Tests for gxe_blup_model.gxe_blup_model."""

from morie.fn import _array_core as np

import math

from morie.fn.gxe_blup_model import gxe_blup_model


def _build_inputs(rng, n=40, p_E=3, n_lines=5, n_envs=2):
    """Build a minimal but consistent set of inputs for ``gxe_blup_model``."""
    # response vector
    y = rng.normal(0, 1, n)

    # environmental design matrix (including intercept)
    X_E = rng.normal(0, 1, (n, p_E))

    # incidence matrices for line and line‑by‑environment effects
    Z_L = np.zeros((n, n_lines))
    Z_EL = np.zeros((n, n_lines * n_envs))

    # assign each observation to a line and an environment
    line_idx = rng.integers(0, n_lines, n)
    env_idx = rng.integers(0, n_envs, n)
    for i in range(n):
        Z_L[i][line_idx[i]] = 1.0
        Z_EL[i][line_idx[i] * n_envs + env_idx[i]] = 1.0

    # genomic relationship matrix (identity is a valid, invertible choice)
    G = np.eye(n_lines)

    # genetic covariance between environments (identity is a valid choice)
    Sigma_E = np.eye(n_envs)

    sigma2_g = 0.5
    sigma2_e = 1.0

    return y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E, sigma2_e


def test_msm018_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E, sigma2_e = _build_inputs(rng)

    result = gxe_blup_model(
        y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E, sigma2_e
    )

    # The returned object must behave like a mapping
    assert isinstance(result, dict)

    # Expected keys
    for key in ("estimate", "beta", "b_lines", "b_gxe", "method"):
        assert key in result

    # Dimensions of the returned arrays:
    # ``beta`` includes the intercept ``mu`` plus the ``p_E`` environmental
    # effects, so its length is ``p_E + 1``.
    assert len(result["beta"]) == len(X_E[0]) + 1
    assert len(result["b_lines"]) == len(Z_L[0])
    assert len(result["b_gxe"]) == len(Z_EL[0])

    # ``estimate`` must be a finite scalar
    assert math.isfinite(result["estimate"])

    # ``method`` must be a string
    assert isinstance(result["method"], str)


def test_msm018_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)

    # Small but still valid configuration
    y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E, sigma2_e = _build_inputs(
        rng, n=8, p_E=1, n_lines=2, n_envs=2
    )

    result = gxe_blup_model(
        y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E, sigma2_e
    )

    assert isinstance(result, dict)
    for key in ("estimate", "beta", "b_lines", "b_gxe", "method"):
        assert key in result

    assert len(result["beta"]) == len(X_E[0]) + 1
    assert len(result["b_lines"]) == len(Z_L[0])
    assert len(result["b_gxe"]) == len(Z_EL[0])

    assert math.isfinite(result["estimate"])
    assert isinstance(result["method"], str)
