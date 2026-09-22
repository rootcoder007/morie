"""Tests for gblpq.gblup_equivalence."""

import numpy as np

from morie.fn import _array_core as _ac
from morie.fn import _frame_core as pd

from morie.fn.gblpq import gblup_equivalence


def test_gblpq_basic():
    """Test basic functionality of the GBLUP Cholesky re-parameterization."""
    rng = np.random.default_rng(43)
    n, q = 20, 4
    Z = rng.normal(0.0, 1.0, (n, q))
    # Construct a positive definite G = B B' + 3*I to exercise a non-trivial Cholesky.
    B = rng.normal(0.0, 1.0, (q, q))
    G = B @ B.T + 3.0 * np.eye(q)
    sigma2_g = 1.7

    result = gblup_equivalence(Z, G, sigma2_g)

    # Documented keys
    for key in ("Zstar", "L", "V_original", "V_reparameterized",
                "max_gap", "n", "q"):
        assert key in result, f"missing key {key!r} in result"

    assert result["n"] == n
    assert result["q"] == q

    # Shapes: Zstar is (n, q); L is (q, q); both variance matrices are (n, n).
    Zstar = np.asarray(result["Zstar"])
    L = np.asarray(result["L"])
    V0 = np.asarray(result["V_original"])
    V1 = np.asarray(result["V_reparameterized"])
    assert Zstar.shape == (n, q)
    assert L.shape == (q, q)
    assert V0.shape == (n, n)
    assert V1.shape == (n, n)

    # G = L L' (lower factor)
    G_arr = np.asarray(G)
    assert np.allclose(L @ L.T, G_arr, atol=1e-10)

    # Zstar = Z L
    Z_arr = np.asarray(Z)
    assert np.allclose(Zstar, Z_arr @ L, atol=1e-10)

    # V_original = sigma2_g * Z G Z'
    expected_V0 = sigma2_g * (Z_arr @ G_arr @ Z_arr.T)
    assert np.allclose(V0, expected_V0, atol=1e-10)

    # V_reparameterized = sigma2_g * Zstar Zstar'
    expected_V1 = sigma2_g * (Zstar @ Zstar.T)
    assert np.allclose(V1, expected_V1, atol=1e-10)

    # max_gap is the largest entry-wise difference between the two marginal variances.
    expected_gap = float(np.max(np.abs(V0 - V1)))
    assert abs(result["max_gap"] - expected_gap) < 1e-9


def test_gblpq_edge():
    """Test edge case: identity G (the simplest positive definite case)."""
    rng = np.random.default_rng(43)
    n, q = 15, 3
    Z = rng.normal(0.0, 1.0, (n, q))
    G = np.eye(q)
    sigma2_g = 2.5

    result = gblup_equivalence(Z, G, sigma2_g)

    assert result["n"] == n
    assert result["q"] == q

    Zstar = np.asarray(result["Zstar"])
    L = np.asarray(result["L"])
    V0 = np.asarray(result["V_original"])
    V1 = np.asarray(result["V_reparameterized"])

    # With G = I_q, the lower Cholesky factor is I_q and Zstar == Z.
    assert np.allclose(L, np.eye(q), atol=1e-12)
    assert np.allclose(Zstar, np.asarray(Z), atol=1e-12)

    # Both marginal variances collapse to sigma2_g * Z Z'.
    expected = sigma2_g * (np.asarray(Z) @ np.asarray(Z).T)
    assert np.allclose(V0, expected, atol=1e-10)
    assert np.allclose(V1, expected, atol=1e-10)
    assert result["max_gap"] < 1e-10
