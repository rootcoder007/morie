"""Tests for otsobm.ot_sobolev_w1."""
import math
import pytest

from morie.fn import _array_core as np
from morie.fn.otsobm import ot_sobolev_w1


def test_otsobm_basic():
    """Test basic functionality."""
    n = 40
    rng = np.random.default_rng(42)
    # Two measures on the same support
    mu = rng.normal(0, 1, n)
    nu = rng.normal(0, 1, n)
    # Build a symmetric inverse Laplacian (positive semi-definite)
    M = rng.normal(0, 1, (n, n))
    Laplace_inv = [[(M[i][j] + M[j][i]) / 2 for j in range(n)] for i in range(n)]

    result = ot_sobolev_w1(mu, nu, Laplace_inv)

    # The function returns a RichResult that behaves like a dict
    assert isinstance(result, dict)
    expected_keys = {"W1_sob", "quad_form", "mass_gap", "n", "method"}
    assert expected_keys.issubset(result.keys())

    # Basic sanity checks
    assert result["n"] == n
    assert math.isfinite(result["W1_sob"])
    assert result["W1_sob"] >= 0.0
    assert result["quad_form"] >= 0.0
    assert math.isfinite(result["mass_gap"])
    assert math.isclose(result["mass_gap"], sum(mu) - sum(nu))


def test_otsobm_edge():
    """Edge case: mu and nu must have the same length."""
    mu = [0.1, 0.2]
    nu = [0.4, 0.5, 0.6]
    # Any Laplacian matrix will do; the length mismatch triggers the ValueError
    Laplace_inv = [[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        ot_sobolev_w1(mu, nu, Laplace_inv)
