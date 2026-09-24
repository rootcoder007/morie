"""Tests for drrct.dr_rct_assisted_did."""

import math

from morie.fn import _array_core as np

from morie.fn.drrct import dr_rct_assisted_did


def test_drrct_basic():
    """Test basic functionality with G=None (every row in both stages)."""
    rng = np.random.default_rng(42)
    n = 60
    p = 3
    y_obs = rng.normal(0, 1, n)
    y_rct = rng.normal(0, 1, n)
    D = [float(x) for x in rng.integers(0, 2, n)]
    # Guarantee both treated and untreated arms are present
    D[0] = 0.0
    D[1] = 1.0
    X = rng.normal(0, 1, (n, p))
    result = dr_rct_assisted_did(y_obs, y_rct, D, X)
    assert isinstance(result, dict)
    for key in ("estimate", "tau_esc", "tau_naive", "correction",
                "delta", "tau_secondary", "alpha_sd",
                "n_exp", "n_obs", "n"):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["tau_esc"])
    assert math.isfinite(result["tau_naive"])
    assert math.isfinite(result["correction"])
    assert math.isfinite(result["delta"])
    assert result["n"] == n
    assert result["n_exp"] == n
    assert result["n_obs"] == n


def test_drrct_edge():
    """Test with explicit experimental/observational subsamples via G."""
    rng = np.random.default_rng(42)
    n = 80
    p = 3
    y_obs = rng.normal(0, 1, n)
    y_rct = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, p))
    # Build G so the first half is experimental (1) and the second half
    # observational (0), with both treatment arms represented in each group.
    G = []
    D = []
    for i in range(n):
        if i < 40:
            G.append(1.0)
        else:
            G.append(0.0)
        D.append(0.0 if (i % 4) < 2 else 1.0)
    result = dr_rct_assisted_did(y_obs, y_rct, D, X, G=G)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "tau_esc" in result
    assert "tau_naive" in result
    assert "correction" in result
    assert "delta" in result
    assert "tau_secondary" in result
    assert "alpha_sd" in result
    assert "n_exp" in result
    assert "n_obs" in result
    assert "n" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["tau_esc"])
    assert result["n"] == n
    assert result["n_exp"] == 40
    assert result["n_obs"] == 40
