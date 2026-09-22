"""Tests for gestid.g_estimation_snm."""

from morie.fn import _array_core as np

from morie.fn.gestid import g_estimation_snm


def test_gestid_basic():
    """Test basic functionality with a known treatment effect."""
    rng = np.random.default_rng(43)
    n = 500
    # X must be shape (n, p) per the docstring
    X = rng.normal(size=(n, 2))
    # d must be binary {0, 1} per the docstring; generate via propensity
    true_beta = rng.normal(size=3)  # intercept + 2 covariates
    B = np.column_stack([np.ones(n), X])
    logits = B @ true_beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    d = (rng.uniform(size=n) < prob).astype(float)
    # Outcome with a known structural parameter psi_true
    psi_true = 1.5
    y = psi_true * d + X[:, 0] + rng.normal(size=n)

    result = g_estimation_snm(y, d, X)

    assert isinstance(result, dict)
    assert "estimate" in result
    # The estimate should be close to the true psi under correct
    # propensity specification (docstring example uses 0.3 tolerance)
    assert abs(result["estimate"] - psi_true) < 0.5


def test_gestid_edge():
    """Test edge cases: custom grid and span arguments."""
    rng = np.random.default_rng(43)
    n = 300
    X = rng.normal(size=(n, 3))
    true_beta = rng.normal(size=4)
    B = np.column_stack([np.ones(n), X])
    logits = B @ true_beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    d = (rng.uniform(size=n) < prob).astype(float)
    psi_true = 0.8
    y = psi_true * d + X[:, 0] + rng.normal(size=n)

    # Custom grid argument: should be a candidate psi array
    custom_grid = np.linspace(-2.0, 3.0, 41)
    result = g_estimation_snm(y, d, X, grid=custom_grid)

    assert isinstance(result, dict)
    assert "test_curve" in result
    # test_curve must be the same length as the supplied grid
    assert len(result["test_curve"]) == len(custom_grid)
    assert np.allclose(result["grid"], custom_grid)
    # estimate should still be near the truth
    assert abs(result["estimate"] - psi_true) < 0.5

    # span argument overrides the automatic grid half-width
    result2 = g_estimation_snm(y, d, X, span=2.0, n_grid=51)
    assert isinstance(result2, dict)
    assert "grid" in result2
    assert len(result2["grid"]) == 51
